import numpy as np

from utility import *


def extract_rect_features(rects, feature_dict, svg_width, svg_height):
    num_rects = 1.0 * len(rects)
    rect_fill_count = {}
    x_list = []
    x_count = {}
    y_list = []
    y_count = {}
    w_list = []
    w_count = {}
    h_list = []
    h_count = {}
    for rect in rects:
        fill = find_attribute(rect, "fill", True, [])
        if is_not_None(fill):
            if rect_fill_count.get(fill):
                rect_fill_count[fill] += 1
            else:
                rect_fill_count[fill] = 1
        h = 0
        w = 0
        for dim, dim_list, dim_dict, dim_var in [("w", w_list, w_count, w), ("h", h_list, h_count, h)]:
            if rect.has_attr(dim):
                dim_var += convert_str_to_float(rect[dim])
            dim_list.append(dim_var)
            if dim_dict.get(dim_var):
                dim_dict[dim_var] += 1
            else:
                dim_dict[dim_var] = 1
        tx, ty = get_location(rect, 1, 0, 0)
        for loc, loc_list, loc_dict, t_var in [("x", x_list, x_count, tx), ("y", y_list, y_count, ty)]:
            loc_var = t_var
            if loc == "y":
                loc_var += h
            if rect.has_attr(loc):
                loc_var += convert_str_to_float(rect[loc])
            loc_list.append(loc_var)
            if loc_dict.get(loc_var):
                loc_dict[loc_var] += 1
            else:
                loc_dict[loc_var] = 1
    feature_dict["colorful_rect_count"] = len(list(rect_fill_count.items())) / num_rects
    for var_list, var_count, name, svg_dim in [(x_list, x_count, "x", svg_width), (y_list, y_count, "y", svg_height)]:
        if len(var_list) > 0:
            np_array = np.array(var_list)
            feature_dict["rect_var_" + name] = np.var(np_array) / svg_dim ** 2
            feature_dict["rect_max_" + name] = np.max(np_array) / svg_dim
            feature_dict["rect_min_" + name] = np.min(np_array) / svg_dim
            feature_dict["rect_num_unique_" + name] = len(list(var_count.keys())) / num_rects
            val_np_array = np.array(list(var_count.values()))
            feature_dict["rect_avg_" + name + "_bin_size"] = np.mean(val_np_array) / num_rects
            feature_dict["rect_var_" + name + "_bin_size"] = np.var(val_np_array) / num_rects ** 2
    for loc_list, loc_count, loc_name, svg_dim in [(w_list, w_count, "w", svg_width),
                                                   (h_list, h_count, "h", svg_height)]:
        if len(loc_list) > 0:
            np_array = np.array(loc_list)
            feature_dict["rect_var_" + loc_name] = np.var(np_array) / svg_dim ** 2
            feature_dict["rect_max_" + loc_name] = np.max(np_array) / svg_dim
            feature_dict["rect_min_" + loc_name] = np.min(np_array) / svg_dim
            feature_dict["rect_num_unique_" + loc_name] = len(list(loc_count.keys())) / num_rects
            val_np_array = np.array(list(loc_count.values()))
            feature_dict["rect_avg_" + loc_name + "_bin_size"] = np.mean(val_np_array) / num_rects
            feature_dict["rect_var_" + loc_name + "_bin_size"] = np.var(val_np_array) / num_rects ** 2
            feature_dict["rect_max_" + loc_name + "_count"] = np.max(val_np_array) / num_rects
            feature_dict["rect_unique_" + loc_name + "_count"] = len(list(loc_count.keys())) / num_rects
    if len(rect_fill_count) > 0:
        fill_list = np.array(list(rect_fill_count.values()))
        feature_dict["rect_avg_fill_bin_size"] = np.mean(fill_list) / num_rects
        feature_dict["rect_var_fill_bin_size"] = np.var(fill_list) / num_rects ** 2
    return feature_dict
