# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
#import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

#from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    #def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
    #def __init__(self, cost_class: float = 1, cost_kpts: float = 1, cost_kpts_class: float = 1):
    def __init__(self, cost_class: float = 1, cost_kpts: float = 1, cost_ctrs: float = 1,
                cost_deltas: float = 1, cost_kpts_class: float = 1, oks = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_kpts: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
            cost_kpts_class: This is the relative weight of the keypoints classification error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        """
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        """
        self.cost_kpts = cost_kpts
        self.cost_ctrs = cost_ctrs
        self.cost_deltas = cost_deltas
        self.cost_kpts_class = cost_kpts_class
        assert cost_class != 0 or cost_kpts != 0 or cost_kpts_class != 0, "all costs cant be 0"
        self.oks = oks

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_kpts": Tensor of dim [batch_size, num_queries, 36] with the predicted center and offsets
                 ("pred_kpts_classes": Tensor of dim [batch_size, num_queries, 17, 34] with the keypoints classes)

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_people] (where num_people is the number of ground-truth
                           people in the target) containing the class labels
                 "keypoints": Tensor of dim [num_people, 36] containing the target center and offsets
                 "keypoints_classes": Tensor of dim [num_people, 34] containing the target center and offsets

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_people)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        #out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_kpts = outputs["pred_kpts"].flatten(0, 1)  # [batch_size * num_queries, 53]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        #tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_kpts = torch.cat([v["keypoints"] for v in targets])

        if self.oks:
            #img_sizes = torch.cat([torch.tensor([v['size'].tolist()]) for v in targets])
            img_sizes = []
            for v in targets:
                for _ in range(len(v['labels'])):
                    img_sizes.append(torch.tensor([v['size'].tolist()]).cuda())
            img_sizes = torch.cat(img_sizes)
            bb_areas = torch.cat([v["area"] for v in targets])

        num_persons = tgt_kpts.shape[0]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        #out_kpts_ids = torch.sigmoid(out_kpts[:, 4::3])
        out_kpts_ids = out_kpts[:, 4::3]
        tgt_kpts_ids = tgt_kpts[:, 5::3].clone()

        x_kpts = torch.cat((out_kpts[:, 0].unsqueeze(-1), out_kpts[:, 2::3]), dim=1)
        y_kpts = torch.cat((out_kpts[:, 1].unsqueeze(-1), out_kpts[:, 3::3]), dim=1)

        # Preparing keypoints for computing matching loss on absolute positions
        x_kpts_abs, y_kpts_abs = x_kpts.clone(), y_kpts.clone()
        x_kpts_abs[:, 1:] = (x_kpts_abs[:, 1:] - 0.5) * 2
        y_kpts_abs[:, 1:] = (y_kpts_abs[:, 1:] - 0.5) * 2
        x_kpts_abs[:, 1:] += x_kpts_abs[:, 0].clone().unsqueeze_(1)
        y_kpts_abs[:, 1:] += y_kpts_abs[:, 0].clone().unsqueeze_(1)
        tgt_kpts_abs = tgt_kpts.clone()
        tgt_kpts_abs[:, 3::3] = (tgt_kpts_abs[:, 3::3] - 0.5) * 2
        tgt_kpts_abs[:, 4::3] = (tgt_kpts_abs[:, 4::3] - 0.5) * 2
        tgt_kpts_abs[:, 3::3] += tgt_kpts_abs[:, 0].clone().unsqueeze_(1)
        tgt_kpts_abs[:, 4::3] += tgt_kpts_abs[:, 1].clone().unsqueeze_(1)

        '''
        if not any(tgt_kpts[:,0:3].clone().flatten()): # centers are learnt
            # transform output to absolute values (adding center)
            x_kpts[:, 1:] = x_kpts[:, 1:] + x_kpts[:, 0].clone().unsqueeze_(1)
            y_kpts[:, 1:] = y_kpts[:, 1:] + y_kpts[:, 0].clone().unsqueeze_(1)
            visible_mask = (tgt_kpts[:, 2::3].clone() == 1)
            out_kpts = torch.stack((x_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                    y_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                    #dim=2).reshape(-1, 36, num_persons)
                                    dim=2).reshape(bs*num_queries, 36, num_persons)
            tgt_kpts = torch.stack((tgt_kpts[:, 0::3].clone() * visible_mask,
                                    tgt_kpts[:, 1::3].clone() * visible_mask), dim=2).view(-1, 36)
            # Compute the L1 cost between keypoints positions
            cost_kpts = torch.empty(x_kpts.shape[0], num_persons).cuda()
            # Compute the L2 cost between center keypoints (for learnt centers always 0)
            cost_ctrs = torch.zeros(x_kpts.shape[0], num_persons).cuda()
            for i in range(num_persons):
                if not self.oks:
                    cost_kpts[:, i, None] = torch.cdist(out_kpts[:,2:,i], tgt_kpts[i,2:].unsqueeze(0), p=1)
            # Compute the L2 cost between keypoints classes.
            cost_kpts_class = torch.cdist(out_kpts_ids, tgt_kpts_ids, p=2)
            #cost_kpts_class = F.cross_entropy(out_kpts_ids, tgt_kpts_ids)
            #cost_kpts_class = nn.BCELoss(out_kpts_ids, tgt_kpts_ids.bool())
        '''
            
        #elif all(tgt_kpts[:,2].clone().flatten()): # all centers are visible (e.g. center of mass)
        if all(tgt_kpts[:,2].clone().flatten()): # all centers are visible (e.g. center of mass)
            visible_mask = (tgt_kpts[:, 2::3].clone() == 1)
            out_kpts = torch.stack((x_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                    y_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                    #dim=2).reshape(-1, 36, num_persons)
                                    dim=2).reshape(bs*num_queries, 36, num_persons)
            tgt_kpts = torch.stack((tgt_kpts[:, 0::3].clone() * visible_mask,
                                    tgt_kpts[:, 1::3].clone() * visible_mask), dim=2).view(-1, 36)
            out_kpts_abs = torch.stack((x_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                        y_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                        dim=2).reshape(bs*num_queries, 36, num_persons)
            tgt_kpts_abs = torch.stack((tgt_kpts_abs[:, 0::3].clone() * visible_mask,
                                        tgt_kpts_abs[:, 1::3].clone() * visible_mask), dim=2).view(-1, 36)
                                
            cost_kpts = torch.empty(x_kpts.shape[0], num_persons).cuda() # L1 cost on absolute positions
            cost_ctrs = torch.empty(x_kpts.shape[0], num_persons).cuda() # L2 cost on centers
            cost_deltas = torch.empty(x_kpts.shape[0], num_persons).cuda() # L1 cost on deltas
            for i in range(num_persons):
                cost_deltas[:, i, None] = torch.cdist(out_kpts[:,2:,i], tgt_kpts[i,2:].unsqueeze(0), p=1)
                #cost_deltas[:, i, None] = torch.cdist(out_kpts[:,:,i], tgt_kpts[i,:].unsqueeze(0), p=1)
                if not self.oks:
                    cost_kpts[:, i, None] = torch.cdist(out_kpts_abs[:,2:,i], tgt_kpts_abs[i,2:].unsqueeze(0), p=1)
                cost_ctrs[:, i, None] = torch.cdist(out_kpts[:,:2,i], tgt_kpts[i,:2].unsqueeze(0), p=2)

            # Compute the L2 cost between keypoints classes
            cost_kpts_class = torch.cdist(out_kpts_ids, tgt_kpts_ids, p=2)
            #cost_kpts_class = F.cross_entropy(out_kpts_ids, tgt_kpts_ids)
            #cost_kpts_class = nn.BCELoss(out_kpts_ids, tgt_kpts_ids.bool())
            
        else:
            #have mask on predicted keypoints (and keypoints classes) when target keypoints are invisible
            visible_mask = (tgt_kpts[:, 2::3].clone() == 1)
            visible_mask[torch.where(visible_mask[:, 0] == 0)] = False
            out_kpts = torch.stack((x_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                    y_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                    #dim=2).reshape(-1, 36, num_persons)
                                    dim=2).reshape(bs*num_queries, 36, num_persons)
            tgt_kpts = torch.stack((tgt_kpts[:, 0::3].clone() * visible_mask,
                                    tgt_kpts[:, 1::3].clone() * visible_mask), dim=2).view(-1, 36)
            out_kpts_abs = torch.stack((x_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                        y_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                        dim=2).reshape(bs*num_queries, 36, num_persons)
            tgt_kpts_abs = torch.stack((tgt_kpts_abs[:, 0::3].clone() * visible_mask,
                                        tgt_kpts_abs[:, 1::3].clone() * visible_mask), dim=2).view(-1, 36)

            src_kpts_ids = out_kpts_ids.clone()
            target_kpts_ids = tgt_kpts_ids.clone()
            src_kpts_ids[torch.where(visible_mask[:, 0] == 0)] = 0
            target_kpts_ids[torch.where(visible_mask[:, 0] == 0)] = 0

            cost_kpts = torch.empty(x_kpts.shape[0], num_persons).cuda() # L1 cost on absolute positions
            cost_ctrs = torch.empty(x_kpts.shape[0], num_persons).cuda() # L2 cost on centers
            cost_deltas = torch.empty(x_kpts.shape[0], num_persons).cuda() # L1 cost on deltas
            cost_kpts_class = torch.empty(x_kpts.shape[0], num_persons).cuda() # L2 cost
            for i in range(num_persons):
                # L1 cost of keypoint offsets
                cost_deltas[:, i, None] = torch.cdist(out_kpts[:,2:,i], tgt_kpts[i,2:].unsqueeze(0), p=1)
                #cost_deltas[:, i, None] = torch.cdist(out_kpts[:,:,i], tgt_kpts[i,:].unsqueeze(0), p=1)
                # L1 cost of keypoints
                if not self.oks:
                    cost_kpts[:, i, None] = torch.cdist(out_kpts_abs[:,2:,i], tgt_kpts_abs[i,2:].unsqueeze(0), p=1)
                # L2 cost of center keypoints
                cost_ctrs[:, i, None] = torch.cdist(out_kpts[:,:2,i], tgt_kpts[i,:2].unsqueeze(0), p=2)
                # L2 cost of keypoint classes
                cost_kpts_class[:, i, None] = torch.cdist(src_kpts_ids, target_kpts_ids[i,:].unsqueeze(0), p=2)
                #cost_kpts_class[:, i, None] = F.cross_entropy(src_kpts_ids, target_kpts_ids[i,:].unsqueeze(0))
                #cost_kpts_class[:, i, None] = nn.BCELoss(src_kpts_ids, target_kpts_ids[i,:].unsqueeze(0).bool())


        if self.oks:
            #tgt_kpts = tgt_kpts.clone()
            #out_kpts = out_kpts.clone()
            #cost_kpts = torch.empty(x_kpts.shape[0], num_persons).cuda()

            # OKS is computed on absolute image coordinates, so we transform back the normalized coordinates
            img_h, img_w = img_sizes.unbind(1)
            out_kpts[..., 0::2, :] = out_kpts_abs[..., 0::2, :].clone() * img_w[(None,)*2 + (...,)]
            out_kpts[..., 1::2, :] = out_kpts_abs[..., 1::2, :].clone() * img_h[(None,)*2 + (...,)]
            tgt_kpts[..., 0::2] = tgt_kpts_abs[..., 0::2].clone() * img_w[(...,) + (None,)]
            tgt_kpts[..., 1::2] = tgt_kpts_abs[..., 1::2].clone() * img_h[(...,) + (None,)]

            sigmas = torch.tensor([.26,.25,.25,.35,.35,.79,.79,.72,.72,.62,.62,1.07,1.07,.87,.87,.89,.89])/10.0
            vars = ((sigmas * 2)**2).cuda()
            
            #for j in range(len(out_kpts)):
            for i in range(num_persons):
                xg = tgt_kpts[i,2::2]; yg = tgt_kpts[i,3::2]; vg = visible_mask[i,1:]
                k1 = vg.sum() # nr of visible keypoints
                xd = out_kpts[:,2::2,i]; yd = out_kpts[:,3::2,i]
                # measure the per-keypoint distance
                dx = xd - xg
                dy = yd - yg
                e = (dx**2 + dy**2) / vars / (bb_areas[i] + torch.finfo(torch.float64).eps) / 2
                if k1 > 0:
                    e = e[:, torch.where(vg > 0)[0]]
                ious = (torch.exp(-e)).sum(dim=1) / e.shape[1]
                cost_kpts[:, i, None] = torch.cdist(torch.ones(1,len(ious)).cuda(), ious.unsqueeze(0), p=2)
            
            '''
            for j in range(len(out_kpts)):
                for i in range(num_persons):
                    xg = tgt_kpts[i,2::2]; yg = tgt_kpts[i,3::2]; vg = visible_mask[i,1:]
                    k1 = vg.sum() # nr of visible keypoints
                    xd = out_kpts[j,2::2,i]; yd = out_kpts[j,3::2,i]
                    # measure the per-keypoint distance
                    dx = xd - xg
                    dy = yd - yg
                    e = (dx**2 + dy**2) / vars / (bb_areas[i] + torch.finfo(torch.float64).eps) / 2            
                    if k1 > 0:
                        e[torch.where(vg > 0)]
                    iou = (torch.exp(-e)).sum() / e.shape[0]
                    cost_kpts[j, i] = (torch.ones(1).cuda() - iou)**2
                    #cost_kpts[j, i] = F.mse_loss(torch.ones(1), iou)
                    #cost_kpts[j, i] = torch.cdist(torch.ones(1), iou , p=2)
                    #cost_kpts[j, i, None] = torch.cdist(out_kpts[j,2:,i], tgt_kpts[i,2:].unsqueeze(0), p=2)
            '''

        # Compute the L1 cost between boxes
        #cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        #cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        #C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        #C = self.cost_kpts * cost_kpts + self.cost_class * cost_class + self.cost_kpts_class * cost_kpts_class
        #C = self.cost_kpts * cost_kpts + self.cost_ctrs * cost_ctrs + self.cost_class * cost_class + self.cost_kpts_class * cost_kpts_class
        C = self.cost_kpts * cost_kpts + self.cost_ctrs * cost_ctrs + self.cost_deltas * cost_deltas + self.cost_class * cost_class + self.cost_kpts_class * cost_kpts_class
        C = C.view(bs, num_queries, -1).cpu()

        #sizes = [len(v["boxes"]) for v in targets]
        sizes = [len(v["keypoints"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    #return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
    return HungarianMatcher(cost_class=args.set_cost_class, cost_kpts=args.set_cost_kpts, cost_ctrs=args.set_cost_ctrs,
                            cost_deltas=args.set_cost_deltas, cost_kpts_class=args.set_cost_kpts_class, oks=args.oks_loss)
