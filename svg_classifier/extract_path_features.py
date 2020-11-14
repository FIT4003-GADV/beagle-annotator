from utility import *
import numpy as np
import re
import sys

def extract_path_features(paths, feature_dict, svg_width, svg_height):
    all_paths_count = 1.0*len(paths)
    filled_path_count = {}
    path_stroke_count = {}
    max_stroke_width = 0
    min_stroke_width = sys.float_info.max
    max_y_distance = 0
    max_x_distance = 0
    max_displacement = 0
    path_d_lengths = []
    x_list = []
    y_list = []
    x_count = {}
    y_count = {}
    z_lengths = []
    rx_radii = []
    ry_radii = []
    circular_a_count_per_element = []
    noncircular_a_count_per_element = []
    q_count_per_element = []
    for path in paths:
        has_fill = False
        attr_var_dict = get_attributes(path)
        for (attr_name, count_dict) in [('fill', filled_path_count), ('stroke', path_stroke_count)]:
            attr = attr_var_dict[attr_name]
            if attr:
                if count_dict.get(attr):
                    count_dict[attr] += 1
                else:
                    count_dict[attr] = 1
                if attr_name == "fill":
                    has_fill = True
        stroke_width = attr_var_dict['stroke-width']
        if stroke_width:
            stroke_width = convert_str_to_float(stroke_width)
            if stroke_width < min_stroke_width:
                min_stroke_width = stroke_width
            if stroke_width > max_stroke_width:
                max_stroke_width = stroke_width
        tx, ty = get_location(path, 1, 0, 0)
        d = path["d"].lower()
        path_d_lengths.append(len(d))
        path_numbers = get_numbers_from_path(d)
        if len(path_numbers) > 3:
            start_x =convert_str_to_float(path_numbers[0]) + tx
            start_y =convert_str_to_float(path_numbers[1]) + ty
            end_x =convert_str_to_float(path_numbers[-2]) + tx
            end_y =convert_str_to_float(path_numbers[-1]) + ty
            for (start, end, max_distance, start_list, count_dict) in [(start_x, end_x, max_x_distance, x_list, x_count), (start_y, end_y, max_y_distance, y_list, y_count)]:
                start_list.append(start)
                if count_dict.get(start):
                    count_dict[start] += 1
                else:
                    count_dict[start] = 1
                if end - start > max_distance and ("z" not in d or has_fill):
                    max_distance = end - start
            displacement = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
            if displacement > max_displacement and ("z" not in d or has_fill):
                max_displacement = displacement

        a_indices = []
        q_count = 0
        for i in range(len(d)):
            if d[i] == 'a':
                a_indices.append(i)
            elif d[i] == 'q':
                q_count += 1
            elif d[i] == 'z':
                z_lengths.append(len(d))
        q_count_per_element.append(q_count)
        path_rx_radii, path_ry_radii, circular_arc_count, noncircular_arc_count = get_a_arcs(a_indices, d)
        rx_radii.extend(path_rx_radii)
        ry_radii.extend(path_ry_radii)
        circular_a_count_per_element.append(circular_arc_count)
        noncircular_a_count_per_element.append(noncircular_arc_count)
        #if is_hexagon(d):
        #    hexagon_count += 1
    feature_dict["path_filled_path_count"] = len(list(filled_path_count.items()))/all_paths_count
    feature_dict["path_stroke_count"] = len(list(path_stroke_count.items()))/all_paths_count
    feature_dict["path_max_stroke_width"] = max_stroke_width
    if min_stroke_width == sys.float_info.max:
        feature_dict["path_max_stroke_width"] = -1
    else:
        feature_dict["path_min_stroke_width"] = min_stroke_width
    feature_dict["path_unique_fill_color_count"] = len(list(filled_path_count.keys()))/all_paths_count
    feature_dict["path_unique_stroke_color_count"] = len(list(path_stroke_count.keys()))/all_paths_count
    feature_dict["path_max_displacement"] = max_displacement/(svg_height**2 + svg_width**2)**0.5
    if feature_dict["path_filled_path_count"] > 0:
        fill_list = np.array(list(filled_path_count.values()))
        feature_dict["path_fill_avg_bin_size"] = np.mean(fill_list)/all_paths_count
        feature_dict["path_fill_var_bin_size"] = np.var(fill_list)/all_paths_count**2
    if feature_dict["path_stroke_count"] > 0:
        stroke_list = np.array(list(path_stroke_count.values()))
        feature_dict["path_stroke_avg_bin_size"] = np.mean(stroke_list)/all_paths_count
        feature_dict["path_stroke_var_bin_size"] = np.var(stroke_list)/all_paths_count**2
    if len(x_list) > 0:
        x_array = np.array(x_list)
        feature_dict["path_x_variance"] = np.var(x_array)/svg_width**2
        feature_dict["path_min_x_position"] = np.min(x_array)/svg_width
        feature_dict["path_max_x_position"] = np.max(x_array)/svg_width
        feature_dict["path_unique_x_position_count"] = len(list(x_count.keys()))/all_paths_count
        x_count_array = np.array(list(x_count.values()))
        feature_dict["path_avg_x_bin_count"] = np.mean(x_count_array)/all_paths_count
        feature_dict["path_var_x_bin_count"] = np.var(x_count_array)/all_paths_count**2
        feature_dict["path_max_x_distance"] = max_x_distance/svg_width
    if len(y_list) > 0:
        y_array = np.array(y_list)
        feature_dict["path_y_variance"] = np.var(y_array)/svg_height**2
        feature_dict["path_min_y_position"] = np.min(y_array)/svg_height
        feature_dict["path_max_y_position"] = np.max(y_array)/svg_height
        feature_dict["path_unique_y_position_count"] = len(list(y_count.keys()))/all_paths_count
        y_count_array = np.array(list(y_count.values()))
        feature_dict["path_avg_y_bin_count"] = np.mean(y_count_array)/all_paths_count
        feature_dict["path_var_y_bin_count"] = np.var(y_count_array)/all_paths_count**2
        feature_dict["path_max_y_distance"] = max_y_distance/svg_height
    if len(path_d_lengths) > 0:
        d_array = np.array(path_d_lengths)
        feature_dict["path_max_d_length"] = np.max(d_array)
        feature_dict["path_min_d_length"] = np.min(d_array)
        feature_dict["path_avg_d_length"] = np.mean(d_array)
        feature_dict["path_var_d_length"] = np.var(d_array)
    if len(circular_a_count_per_element) > 0:
        circular_a_count_per_element_array = np.array(circular_a_count_per_element)
        feature_dict["path_max_num_circular_arc_per_element"] = np.max(circular_a_count_per_element_array)
        feature_dict["path_min_num_circular_arc_per_element"] = np.min(circular_a_count_per_element_array)
        feature_dict["path_avg_num_circular_arc_per_element"] = np.mean(circular_a_count_per_element_array)
        feature_dict["path_num_circular_arcs"] = np.sum(circular_a_count_per_element_array)
    if len(circular_a_count_per_element) > 0:
        noncircular_a_count_per_element_array = np.array(noncircular_a_count_per_element)
        feature_dict["path_max_num_noncircular_arc_per_element"] = np.max(noncircular_a_count_per_element_array)
        feature_dict["path_min_num_noncircular_arc_per_element"] = np.min(noncircular_a_count_per_element_array)
        feature_dict["path_avg_num_noncircular_arc_per_element"] = np.mean(noncircular_a_count_per_element_array)
        feature_dict["path_num_noncircular_arcs"] = np.sum(noncircular_a_count_per_element_array)
    feature_dict["path_unique_rx_count"] = len(set(rx_radii))/all_paths_count
    feature_dict["path_unique_ry_count"] = len(set(ry_radii))/all_paths_count
    if len(rx_radii) > 0:
        rx_radii_array = np.array(rx_radii)
        feature_dict["path_rx_var"] = np.var(rx_radii_array)/all_paths_count**2
    if len(ry_radii) > 0:
        ry_radii_array = np.array(ry_radii)
        feature_dict["path_ry_var"] = np.var(ry_radii_array)/all_paths_count**2
    if len(q_count_per_element) > 0:
        q_count_per_element_array = np.array(q_count_per_element)
        feature_dict["path_max_num_q_arc_per_element"] = np.max(q_count_per_element)
        feature_dict["path_min_num_q_arc_per_element"] = np.min(q_count_per_element)
        feature_dict["path_avg_num_q_arc_per_element"] = np.mean(q_count_per_element)
        feature_dict["path_num_q_arcs"] = np.sum(q_count_per_element)/all_paths_count
    feature_dict["path_z_count"] = len(z_lengths)/all_paths_count
    if feature_dict["path_z_count"] > 0:
        z_array = np.array(z_lengths)
        feature_dict["path_max_d_length_z"] = np.max(z_array)
        feature_dict["path_min_d_length_z"] = np.min(z_array)
        feature_dict["path_avg_d_length_z"] = np.mean(z_array)
        feature_dict["path_var_d_length_z"] = np.var(z_array)
    return feature_dict


def get_a_arcs(arc_indices, d):
    path_rx_radii = []
    path_ry_radii = []
    circular_arc_count = 0
    noncircular_arc_count = 0
    if len(arc_indices) == 0:
        return (path_rx_radii, path_ry_radii, circular_arc_count, noncircular_arc_count)
    for i in range(len(arc_indices) - 1):
        arc_numbers = get_numbers_from_path(d[arc_indices[i]+1:arc_indices[i+1]])
        if len(arc_numbers) < 7:
            continue
        rx = convert_str_to_float(arc_numbers[0])
        ry = convert_str_to_float(arc_numbers[1])
        if rx == ry:
            circular_arc_count += 1
        else:
            noncircular_arc_count += 1
        path_rx_radii.append(rx)
        path_ry_radii.append(ry)
    arc_numbers = get_numbers_from_path(d[arc_indices[len(arc_indices) - 1]+1:])
    if len(arc_numbers) >= 7:
        rx = convert_str_to_float(arc_numbers[0])
        ry = convert_str_to_float(arc_numbers[1])
        if rx == ry:
            circular_arc_count += 1
        else:
            noncircular_arc_count += 1
        path_rx_radii.append(rx)
        path_ry_radii.append(ry)
    return (path_rx_radii, path_ry_radii, circular_arc_count, noncircular_arc_count)



def get_attributes(path):
    attr_dict = {'stroke': None, 'fill': None, 'stroke-width': None}
    stroke_width = find_attribute(path, 'stroke-width', False, [])
    for attr, include_parent, keywords in [('stroke', True, ['line']), ('fill', True, ['area', 'shape']), ('stroke-width', False, [])]:
        attr_var = find_attribute(path, attr, include_parent, keywords)
        if is_not_None(attr_var):
            attr_dict[attr] = attr_var
    return attr_dict

# determines whether a path trajectory is a hexagon.  The path trajectory must contain l
# elements only and three of the hexagonal sides are identical in length (within a small error bound)
def is_hexagon(d):
    d = d[d.rfind("m"):]
    if len(d) < 12:
        return False
    if d[-1] == 'z':
        alpha_chars = [x for x in d[1:-1] if ord(x) >= 97 and ord(x) <= 122]
        for a_c in alpha_chars:
            if a_c != 'l' and a_c != 'e':
                return False
        coordinates = get_numbers_from_path(d)
        if len(coordinates) < 12:
            return False
        x1 = convert_str_to_float(coordinates[2])
        y1 = convert_str_to_float(coordinates[3])
        x2 = convert_str_to_float(coordinates[4])
        y2 = convert_str_to_float(coordinates[5])
        x3 = convert_str_to_float(coordinates[6])
        y3 = convert_str_to_float(coordinates[7])
        x4 = convert_str_to_float(coordinates[8])
        y4 = convert_str_to_float(coordinates[9])
        x5 = convert_str_to_float(coordinates[10])
        y5 = convert_str_to_float(coordinates[11])
        return abs(distance2(x1,y1,x2,y2) - distance2(x2,y2,x3,y3)) < 0.1*distance2(x1,y1,x2,y2) and abs(distance2(x2,y2,x3,y3) - distance2(x3,y3,x4,y4)) < 0.1*distance2(x2,y2,x3,y3) and abs(distance2(x3,y3,x4,y4) - distance2(x4,y4,x5,y5)) < 0.1*distance2(x3,y3,x4,y4)
    return False



