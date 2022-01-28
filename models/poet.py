# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
POET model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, is_dist_avail_and_initialized)
#from util.misc import interpolate

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class POET(nn.Module):
    """ This is the POET module that performs multi-instance pose estimation """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 53, 3)  # 17 keypoints(x,y,o) + center(x,y)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_kpts": The normalized kpts center+offsets for all queries, represented as
                               (center_x, center_y, x+y offsets). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized keypoints.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.kpt_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_kpts': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_kpts': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for POET.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth keypoints and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_people, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_people]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_people):
        """ Compute the cardinality error, ie the absolute error in the number of predicted present people
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_kpts(self, outputs, targets, indices, num_people):
        """Compute the losses related to the keypoints, the L1 position loss and the L2 class loss
            + L2 position loss for centers only
           -- computing binary cross-entropy loss for classes ?? -- 
           targets dicts must contain the key "keypoints" containing a tensor of dim [num_people, 53]
           The target keypoints are expected normalized by the image size.
        """
        assert 'pred_kpts' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        target_kpts = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        src_kpts = outputs['pred_kpts'][idx].clone()
        x_kpts = torch.cat((src_kpts[:, 0].unsqueeze(-1), src_kpts[:, 2::3]), dim=1)
        y_kpts = torch.cat((src_kpts[:, 1].unsqueeze(-1), src_kpts[:, 3::3]), dim=1)

        # Preparing keypoints for computing loss on absolute positions
        x_kpts_abs, y_kpts_abs = x_kpts.clone(), y_kpts.clone()
        x_kpts_abs[:, 1:] = (x_kpts_abs[:, 1:].clone() - 0.5) * 2
        y_kpts_abs[:, 1:] = (y_kpts_abs[:, 1:].clone() - 0.5) * 2
        x_kpts_abs[:, 1:] += x_kpts_abs[:, 0].clone().unsqueeze_(1)
        y_kpts_abs[:, 1:] += y_kpts_abs[:, 0].clone().unsqueeze_(1)
        target_kpts_abs = target_kpts.clone()
        target_kpts_abs[:, 3::3] = (target_kpts_abs[:, 3::3].clone() - 0.5) * 2
        target_kpts_abs[:, 4::3] = (target_kpts_abs[:, 4::3].clone() - 0.5) * 2
        target_kpts_abs[:, 3::3] += target_kpts_abs[:, 0].clone().unsqueeze_(1)
        target_kpts_abs[:, 4::3] += target_kpts_abs[:, 1].clone().unsqueeze_(1)

        #src_kpts_classes = torch.sigmoid(outputs['pred_kpts'][idx][:, 4::3]).clone()
        src_kpts_classes = outputs['pred_kpts'][idx][:, 4::3].clone()
        target_kpts_classes = target_kpts[:, 5::3].clone()
            
        if all(target_kpts[:,2].clone().flatten()): # all centers are visible (e.g. center of mass)

            visible_mask = (target_kpts[:, 2::3].clone() == 1)
            target_kpts = torch.stack(
                (target_kpts[:, 0::3].clone() * visible_mask, target_kpts[:, 1::3].clone() * visible_mask),
                                      dim=2).view(-1, 36)
            src_kpts = torch.stack((x_kpts * visible_mask, y_kpts * visible_mask), dim=2).view(-1, 36)
            loss_deltas = F.l1_loss(src_kpts[:,2:], target_kpts[:,2:], reduction='none')
            #loss_deltas = F.l1_loss(src_kpts[:,:], target_kpts[:,:], reduction='none') # 'middle' experiment
            
            loss_ctrs = F.mse_loss(src_kpts[:,:2], target_kpts[:,:2], reduction='none')

            target_kpts_abs = torch.stack(
                (target_kpts_abs[:, 0::3].clone() * visible_mask, target_kpts_abs[:, 1::3].clone() * visible_mask),
                                      dim=2).view(-1, 36)
            src_kpts_abs = torch.stack((x_kpts_abs * visible_mask, y_kpts_abs * visible_mask), dim=2).view(-1, 36)
            loss_kpts = F.l1_loss(src_kpts_abs[:,2:], target_kpts_abs[:,2:], reduction='none')


        else:

            visible_mask = (target_kpts[:, 2::3].clone() == 1)
            # don't compute position/kpt class loss if center isn't visible
            visible_mask[torch.where(visible_mask[:, 0] == 0)] = False
            target_kpts = torch.stack(
                (target_kpts[:, 0::3].clone() * visible_mask, target_kpts[:, 1::3].clone() * visible_mask),
                dim=2).view(-1, 36)
            src_kpts = torch.stack((x_kpts * visible_mask, y_kpts * visible_mask), dim=2).view(-1, 36)
            loss_deltas = F.l1_loss(src_kpts[:,2:], target_kpts[:,2:], reduction='none')
            #loss_deltas = F.l1_loss(src_kpts[:,:], target_kpts[:,:], reduction='none') # 'middle' experiment

            loss_ctrs = F.mse_loss(src_kpts[:,:2], target_kpts[:,:2], reduction='none')

            target_kpts_abs = torch.stack(
                (target_kpts_abs[:, 0::3].clone() * visible_mask, target_kpts_abs[:, 1::3].clone() * visible_mask),
                                      dim=2).view(-1, 36)
            src_kpts_abs = torch.stack((x_kpts_abs * visible_mask, y_kpts_abs * visible_mask), dim=2).view(-1, 36)
            loss_kpts = F.l1_loss(src_kpts_abs[:,2:], target_kpts_abs[:,2:], reduction='none')
    
            src_kpts_classes[torch.where(visible_mask[:, 0] == 0)] = 0
            target_kpts_classes[torch.where(visible_mask[:, 0] == 0)] = 0
            
        
        losses = {}
        losses['loss_deltas'] = loss_deltas.sum() / num_people
        losses['loss_ctrs'] = loss_ctrs.sum() / num_people
        losses['loss_kpts'] = loss_kpts.sum() / num_people
            
        #loss_kpts_classes = F.mse_loss(src_kpts_classes, target_kpts_classes)
        loss_kpts_classes = F.mse_loss(src_kpts_classes, target_kpts_classes, reduction='none')
        losses['loss_kpts_class'] = loss_kpts_classes.sum() / num_people

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    #def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
    def get_loss(self, loss, outputs, targets, indices, num_people, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'keypoints': self.loss_kpts
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_people, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target individuals accross all nodes, for normalization purposes
        num_people = sum(len(t["labels"]) for t in targets)
        num_people = torch.as_tensor([num_people], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_people)
        num_people = torch.clamp(num_people / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_people))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_people, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_kpts = outputs['pred_logits'], outputs['pred_kpts']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        #scores, labels = prob[..., 1].max(-1)

        # put offsets back into range [-1,1]
        out_kpts[..., 2::3] = (out_kpts[..., 2::3] - 0.5) * 2
        out_kpts[..., 3::3] = (out_kpts[..., 3::3] - 0.5) * 2

        # convert the center-relative positions to image-absolute positions
        out_kpts[..., 2::3] += out_kpts[..., 0].unsqueeze(-1)
        out_kpts[..., 3::3] += out_kpts[..., 1].unsqueeze(-1)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        out_kpts[..., 2::3] *= img_w[(...,) + (None,) * 2]
        out_kpts[..., 3::3] *= img_h[(...,) + (None,) * 2]
        out_kpts[..., 4::3] = 1 # set visibility to 1 as described in COCO API
        results = [{'scores': s, 'labels': l, 'keypoints': k} for s, l, k
                   in zip(scores, labels, out_kpts[..., 2:])]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2
    
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = POET(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_kpts': args.kpts_loss_coef, 'loss_ctrs': args.ctrs_loss_coef,
                    'loss_deltas': args.deltas_loss_coef, 'loss_kpts_class': args.kpts_class_loss_coef}
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'keypoints', 'cardinality']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    postprocessors = {'kpts': PostProcess()}

    return model, criterion, postprocessors
