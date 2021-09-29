import numpy as np

from utility import *


def extract_line_features(lines, feature_dict, svg_width, svg_height):
    num_lines = 1.0 * len(lines)
    v_h_line_count = 0
    stroke_count = {}
    x_list = []
    x_count = {}
    y_list = []
    y_count = {}
    lengths_list = []
    for line in lines:
        if is_line_vertical_or_horizontal(line):
            v_h_line_count += 1
        # stroke = find_attribute(line, "stroke", True, ["edge"])
        stroke = find_attribute(line, "stroke", True, [])
        if is_not_None(stroke):
            if stroke_count.get(stroke):
                stroke_count[stroke] += 1
            else:
                stroke_count[stroke] = 1
        tx, ty = get_location(line, 1, 0, 0)
        for loc, loc_list, loc_dict, t_var in [("x1", x_list, x_count, tx), ("y1", y_list, y_count, ty)]:
            loc_var = t_var
            if line.has_attr(loc):
                loc_var += convert_str_to_float(line[loc])
            loc_list.append(loc_var)
            if loc_dict.get(loc_var):
                loc_dict[loc_var] += 1
            else:
                loc_dict[loc_var] = 1
        x2 = tx
        y2 = ty
        if line.has_attr("x2"):
            x2 += convert_str_to_float(line["x2"])
        if line.has_attr("y2"):
            y2 += convert_str_to_float(line["y2"])
        lengths_list.append(distance(x_list[-1], y_list[-1], x2, y2))
    feature_dict["line_v_h_count"] = v_h_line_count / num_lines
    for var_list, var_count, name, svg_dim in [(x_list, x_count, "x", svg_width), (y_list, y_count, "y", svg_height)]:
        if len(var_list) > 0:
            np_array = np.array(var_list)
            feature_dict["line_var_" + name] = np.var(np_array) / svg_dim ** 2
            feature_dict["line_max_" + name] = np.max(np_array) / svg_dim
            feature_dict["line_min_" + name] = np.min(np_array) / svg_dim
            feature_dict["line_num_unique_" + name] = len(list(var_count.keys())) / num_lines
            val_np_array = np.array(list(var_count.values()))
            feature_dict["line_avg_" + name + "_bin_size"] = np.mean(val_np_array) / num_lines
            feature_dict["line_var_" + name + "_bin_size"] = np.var(val_np_array) / num_lines ** 2
    if len(stroke_count) > 0:
        stroke_list = np.array(list(stroke_count.values()))
        feature_dict["line_stroke_avg_bin_size"] = np.mean(stroke_list) / num_lines
        feature_dict["line_stroke_var_bin_size"] = np.var(stroke_list) / num_lines ** 2
    if len(lengths_list) > 0:
        lengths_array = np.array(lengths_list)
        diag_length = (svg_height ** 2 + svg_width ** 2) ** 0.5
        feature_dict["line_var_length"] = np.var(lengths_array) / diag_length ** 2
        feature_dict["line_max_length"] = np.max(lengths_array) / diag_length
        feature_dict["line_min_length"] = np.min(lengths_array) / diag_length
    return feature_dict


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
