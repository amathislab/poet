"""
Utilities for keypoints manipulation.
"""

import torch

COCO_CLASSES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

SKELETON = [ [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7] ]

# TODO include VISIBILITY !!!!!
# by looking at visibility mask and asking if joint is visible?
def build_JointDist(kpts):
    distances = torch.zeros(len(kpts), len(SKELETON))
    # kpts should be filled with absolute coordinates
    if len(kpts) == 54:
        # throw out center and visibilities
        coord = kpts[:,3:].reshape(-1,3)[:,:2].reshape(len(kpts),-1)
    elif len(kpts) == 53:
        # throw out center (wo vis) and visibilities
        coord = kpts[:,2:].reshape(-1,3)[:,:2].reshape(len(kpts),-1)
    elif len(kpts) == 36:
        # throw out center coordinates
        coord = kpts[:,2:]
    else: # len(kpts) == 34
        coord = kpts
    for i,(j1,j2) in enumerate(SKELETON):
        distances[:,i] = torch.diag(torch.cdist(coord[:,(j1-1)*2:(j1-1)*2+2],
                                    coord[:,(j2-1)*2:(j2-1)*2+2], p=2))
    return distances

def kpts_cxcydxdy_to_xyxy():
    return -1

def kpts_xyxy_to_cxcydxdy(kpts, ctr, hierarchical=False):
    kpts = build_HSPR(kpts, ctr) if hierarchical else build_SPR(kpts, ctr)
    return torch.cat((ctr, kpts), dim=1)


# Structured Pose Representation (SPR)
def build_SPR(keypoints, center):
    keypoints[:, 0::3] = keypoints[:, 0::3] - center[:, 0].unsqueeze_(1)
    keypoints[:, 1::3] = keypoints[:, 1::3] - center[:, 1].unsqueeze_(1)
    return keypoints


# Hierarchical Structured Pose Representation (HSPR)
def build_HSPR(keypoints, center):
    # hierarchical levels:
    
    #   - center -> nose -> eyes, ears
    #keypoints[:, 0:3].clone() # nose
    #keypoints[:, 3:6].clone() # left eye
    #keypoints[:, 6:9].clone() # right eye
    #keypoints[:, 9:12].clone() # left ear
    #keypoints[:, 12:15].clone() # right ear
    nose = keypoints[:, COCO_CLASSES.index("nose")*3:COCO_CLASSES.index("nose")*3+3].clone()
    keypoints[:, 3:15:3] = keypoints[:, 3:15:3] - nose[:, 0].unsqueeze_(1) # eyes + ears
    keypoints[:, 4:15:3] = keypoints[:, 4:15:3] - nose[:, 1].unsqueeze_(1) # eyes + ears
    keypoints[:, 0:3:3] = keypoints[:, 0:3:3] - center[:, 0].unsqueeze_(1) # nose
    keypoints[:, 1:3:3] = keypoints[:, 1:3:3] - center[:, 1].unsqueeze_(1) # nose
    
    #   - center -> shoulders -> elbows -> wrists
    #keypoints[:, 15:18].clone() # left shoulder
    #keypoints[:, 18:21].clone() # right shoulder
    #keypoints[:, 21:24].clone() # left elbow
    #keypoints[:, 24:27].clone() # right elbow
    #keypoints[:, 27:30].clone() # left wrist
    #keypoints[:, 30:33].clone() # right wrist
    l_elbow = keypoints[:, COCO_CLASSES.index("left_elbow")*3:COCO_CLASSES.index("left_elbow")*3+3].clone()
    keypoints[:, 27:30:3] = keypoints[:, 27:30:3] - l_elbow[:, 0].unsqueeze_(1) # left wrist
    keypoints[:, 28:30:3] = keypoints[:, 28:30:3] - l_elbow[:, 1].unsqueeze_(1) # left wrist
    r_elbow = keypoints[:, COCO_CLASSES.index("right_elbow")*3:COCO_CLASSES.index("right_elbow")*3+3].clone()
    keypoints[:, 30:30:3] = keypoints[:, 30:33:3] - r_elbow[:, 0].unsqueeze_(1) # right wrist
    keypoints[:, 31:30:3] = keypoints[:, 31:33:3] - r_elbow[:, 1].unsqueeze_(1) # right wrist
    l_shoulder = keypoints[:, COCO_CLASSES.index("left_shoulder")*3:COCO_CLASSES.index("left_shoulder")*3+3].clone()
    keypoints[:, 21:24:3] = keypoints[:, 21:24:3] - l_shoulder[:, 0].unsqueeze_(1) # left elbow
    keypoints[:, 22:24:3] = keypoints[:, 22:24:3] - l_shoulder[:, 1].unsqueeze_(1) # left elbow
    r_shoulder = keypoints[:, COCO_CLASSES.index("right_shoulder")*3:COCO_CLASSES.index("right_shoulder")*3+3].clone()
    keypoints[:, 24:27:3] = keypoints[:, 25:27:3] - r_shoulder[:, 0].unsqueeze_(1) # right elbow
    keypoints[:, 25:27:3] = keypoints[:, 25:27:3] - r_shoulder[:, 1].unsqueeze_(1) # right elbow
    keypoints[:, 15:18:3] = keypoints[:, 15:18:3] - center[:, 0].unsqueeze_(1) # left shoulder
    keypoints[:, 16:18:3] = keypoints[:, 16:18:3] - center[:, 1].unsqueeze_(1) # left shoulder
    keypoints[:, 18:21:3] = keypoints[:, 18:21:3] - center[:, 0].unsqueeze_(1) # right shoulder
    keypoints[:, 19:21:3] = keypoints[:, 19:21:3] - center[:, 1].unsqueeze_(1) # right shoulder
    
    #   - center -> hips -> knees -> ankles
    #keypoints[:, 33:36].clone() # left hip
    #keypoints[:, 36:39].clone() # right hip
    #keypoints[:, 39:42].clone() # left knee
    #keypoints[:, 42:45].clone() # right knee
    #keypoints[:, 45:48].clone() # left ankle
    #keypoints[:, 48:51].clone() # right ankle
    l_knee = keypoints[:, COCO_CLASSES.index("left_knee")*3:COCO_CLASSES.index("left_knee")*3+3].clone()
    keypoints[:, 45:48:3] = keypoints[:, 45:48:3] - l_knee[:, 0].unsqueeze_(1) # left ankle
    keypoints[:, 46:48:3] = keypoints[:, 46:48:3] - l_knee[:, 1].unsqueeze_(1) # left ankle
    r_knee = keypoints[:, COCO_CLASSES.index("right_knee")*3:COCO_CLASSES.index("right_knee")*3+3].clone()
    keypoints[:, 48:51:3] = keypoints[:, 48:51:3] - r_knee[:, 0].unsqueeze_(1) # right ankle
    keypoints[:, 49:51:3] = keypoints[:, 49:51:3] - r_knee[:, 1].unsqueeze_(1) # right ankle
    l_hip = keypoints[:, COCO_CLASSES.index("left_hip")*3:COCO_CLASSES.index("left_hip")*3+3].clone()
    keypoints[:, 39:42:3] = keypoints[:, 39:42:3] - l_hip[:, 0].unsqueeze_(1) # left knee
    keypoints[:, 40:42:3] = keypoints[:, 40:42:3] - l_hip[:, 1].unsqueeze_(1) # left knee
    r_hip = keypoints[:, COCO_CLASSES.index("right_hip")*3:COCO_CLASSES.index("right_hip")*3+3].clone()
    keypoints[:, 42:45:3] = keypoints[:, 42:45:3] - r_hip[:, 0].unsqueeze_(1) # right knee
    keypoints[:, 43:45:3] = keypoints[:, 43:45:3] - r_hip[:, 1].unsqueeze_(1) # right knee
    keypoints[:, 33:36:3] = keypoints[:, 33:36:3] - center[:, 0].unsqueeze_(1) # left hip
    keypoints[:, 34:36:3] = keypoints[:, 34:36:3] - center[:, 1].unsqueeze_(1) # left hip
    keypoints[:, 36:39:3] = keypoints[:, 36:39:3] - center[:, 0].unsqueeze_(1) # right hip
    keypoints[:, 37:39:3] = keypoints[:, 37:39:3] - center[:, 1].unsqueeze_(1) # right hip
    
    return keypoints

