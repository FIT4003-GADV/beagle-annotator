import numpy as np

from utility import *


def extract_circle_features(circles, feature_dict, svg_width, svg_height):
    num_circles = 1.0 * len(circles)
    circle_fill_count = {}
    x_list = []
    x_count = {}
    y_list = []
    y_count = {}
    radii_list = []
    radii_count = {}
    for circle in circles:
        fill = find_attribute(circle, "fill", True, [])
        if is_not_None(fill):
            if circle_fill_count.get(fill):
                circle_fill_count[fill] += 1
            else:
                circle_fill_count[fill] = 1
        tx, ty = get_location(circle, 1, 0, 0)
        for dim, dim_list, dim_dict, t_var in [("cx", x_list, x_count, tx), ("cy", y_list, y_count, ty)]:
            dim_var = t_var
            if circle.has_attr(dim):
                dim_var += convert_str_to_float(circle[dim])
            dim_list.append(dim_var)
            if dim_dict.get(dim_var):
                dim_dict[dim_var] += 1
            else:
                dim_dict[dim_var] = 1
        r = convert_str_to_float(circle["r"])
        radii_list.append(r)
        if r in list(radii_count.keys()):
            radii_count[r] += 1
        else:
            radii_count[r] = 1
    for var_list, var_count, name, svg_dim in [(x_list, x_count, "x", svg_width), (y_list, y_count, "y", svg_height)]:
        if len(var_list) > 0:
            np_array = np.array(var_list)
            feature_dict["circle_var_" + name] = np.var(np_array) / svg_dim ** 2
            feature_dict["circle_max_" + name] = np.max(np_array) / svg_dim
            feature_dict["circle_min_" + name] = np.min(np_array) / svg_dim
            feature_dict["circle_num_unique_" + name + "_count"] = len(list(var_count.keys())) / num_circles
            val_np_array = np.array(list(var_count.values()))
            feature_dict["circle_avg_" + name + "_bin_size"] = np.mean(val_np_array) / num_circles
            feature_dict["circle_var_" + name + "_bin_size"] = np.var(val_np_array) / num_circles ** 2
    if len(circle_fill_count) > 0:
        fill_list = np.array(list(circle_fill_count.values()))
        feature_dict["circle_avg_fill_bin_size"] = np.mean(fill_list) / num_circles
        feature_dict["circle_var_fill_bin_size"] = np.var(fill_list) / num_circles ** 2
    max_dim = max(svg_width, svg_height)
    if len(radii_list) > 0:
        radii_array = np.array(radii_list)
        feature_dict["circle_min_radius"] = np.min(radii_array) / max_dim
        feature_dict["circle_max_radius"] = np.max(radii_array) / max_dim
        feature_dict["circle_var_radius"] = np.var(radii_array) / max_dim ** 2
        feature_dict["circle_max_radii_count"] = max(radii_count.values()) / num_circles
    return feature_dict
