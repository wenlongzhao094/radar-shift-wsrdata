# The default target_scale_factor is average sheldon factor
def scale_XYWH_box(bbox, dim, annotator_scale_factor, target_scale_factor=0.7429):
    radius = [bbox[2] / 2, bbox[3] / 2]
    new_radius = [r / annotator_scale_factor * target_scale_factor for r in radius]
    center = [(2 * bbox[0] + bbox[2]) / 2, (2 * bbox[1] + bbox[3]) / 2]

    new_left = max(0, center[0] - new_radius[0])
    new_right = min(dim - 1, center[0] + new_radius[0])
    new_top = max(0, center[1] - new_radius[1])
    new_bottom = min(dim - 1, center[1] + new_radius[1])
    return [new_left, new_top, new_right - new_left, new_bottom - new_top]