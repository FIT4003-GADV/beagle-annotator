from bs4 import BeautifulSoup

from axes import *
from extract_circle_features import *
from extract_line_features import *
from extract_path_features import *
from extract_rect_features import *
from extract_text_features import *
from utility import *


# Extracts features from svg file, populates the feature dictionary
# based on a precedence ordering, and returns the feature dictionary

def extract(filename):
    soup = BeautifulSoup(open(filename, 'r').read(), "html.parser")
    if len(soup.prettify()) == 0:
        return "Empty File"
    feature_dict = {
        "path_filled_path_count": 0,
        "path_max_stroke_width": 0,
        "path_min_stroke_width": -1,
        "path_unique_fill_color_count": 0,
        "path_unique_stroke_color_count": 0,
        "path_x_variance": 0,
        "path_y_variance": 0,
        "path_min_x_position": -1,
        "path_min_y_position": -1,
        "path_max_x_position": 0,
        "path_max_y_position": 0,
        "path_unique_x_position_count": 0,
        "path_unique_y_position_count": 0,
        "path_avg_x_bin_count": 0,
        "path_avg_y_bin_count": 0,
        "path_var_x_bin_count": 0,
        "path_var_y_bin_count": 0,
        "path_max_x_distance": 0,
        "path_max_y_distance": 0,
        "path_max_d_length": 0,
        "path_min_d_length": -1,
        "path_avg_d_length": 0,
        "path_var_d_length": 0,
        "path_max_num_circular_arc_per_element": 0,
        "path_min_num_circular_arc_per_element": -1,
        "path_avg_num_circular_arc_per_element": 0,
        "path_num_circular_arcs": 0,
        "path_max_num_noncircular_arc_per_element": 0,
        "path_min_num_noncircular_arc_per_element": -1,
        "path_avg_num_noncircular_arc_per_element": 0,
        "path_num_noncircular_arcs": 0,
        "path_unique_rx_count": 0,
        "path_unique_ry_count": 0,
        "path_rx_var": 0,
        "path_ry_var": 0,
        "path_max_num_q_arc_per_element": 0,
        "path_min_num_q_arc_per_element": -1,
        "path_avg_num_q_arc_per_element": 0,
        "path_num_q_arcs": 0,
        "path_z_count": 0,
        "path_max_d_length_z": 0,
        "path_min_d_length_z": -1,
        "path_avg_d_length_z": 0,
        "path_var_d_length_z": 0,
        "path_hexagon_count": 0,
        "path_class_count": 0,
        "path_stroke_avg_bin_size": 0,
        "path_stroke_var_bin_size": 0,
        "path_fill_avg_bin_size": 0,
        "path_fill_var_bin_size": 0,
        "colorful_circle_count": 0,
        "circle_unique_fill_color_count": 0,
        "circle_var_x": 0,
        "circle_max_x": 0,
        "circle_min_x": -1,
        "circle_num_unique_x_count": 0,
        "circle_avg_x_bin_size": 0,
        "circle_var_x_bin_size": 0,
        "circle_var_y": 0,
        "circle_max_y": 0,
        "circle_min_y": -1,
        "circle_num_unique_y_count": 0,
        "circle_avg_y_bin_size": 0,
        "circle_var_y_bin_size": 0,
        "circle_min_radius": -1,
        "circle_max_radius": 0,
        "circle_var_radius": 0,
        "circle_max_radii_count": 0,
        "circle_class_count": 0,
        "circle_avg_fill_bin_size": 0,
        "circle_var_fill_bin_size": 0,
        "colorful_rect_count": 0,
        "rect_unique_fill_color_count": 0,
        "rect_var_x": 0,
        "rect_max_x": 0,
        "rect_min_x": -1,
        "rect_num_unique_x": 0,
        "rect_avg_x_bin_size": 0,
        "rect_var_x_bin_size": 0,
        "rect_var_y": 0,
        "rect_max_y": 0,
        "rect_min_y": -1,
        "rect_num_unique_y": 0,
        "rect_avg_y_bin_size": 0,
        "rect_var_y_bin_size": 0,
        "rect_var_w": 0,
        "rect_max_w": 0,
        "rect_min_w": -1,
        "rect_num_unique_w": 0,
        "rect_avg_w_bin_size": 0,
        "rect_var_w_bin_size": 0,
        "rect_max_w_count": 0,
        "rect_unique_w_count": 0,
        "rect_var_h": 0,
        "rect_max_h": 0,
        "rect_min_h": -1,
        "rect_num_unique_h": 0,
        "rect_avg_h_bin_size": 0,
        "rect_var_h_bin_size": 0,
        "rect_max_h_count": 0,
        "rect_unique_h_count": 0,
        "rect_class_count": 0,
        "rect_avg_fill_bin_size": 0,
        "rect_var_fill_bin_size": 0,
        "text_word_count": 0,
        "text_max_font_size": 0,
        "text_min_font_size": -1,
        "text_var_font_size": 0,
        "text_unique_font_size_count": 0,
        "line_unique_stroke_color": 0,
        "text_unique_x_count": 0,
        "text_unique_y_count": 0,
        "line_v_h_count": 0,
        "line_var_x": 0,
        "line_max_x": 0,
        "line_min_x": -1,
        "line_num_unique_x": 0,
        "line_avg_x_bin_size": 0,
        "line_var_x_bin_size": 0,
        "line_var_y": 0,
        "line_max_y": 0,
        "line_min_y": -1,
        "line_num_unique_y": 0,
        "line_avg_y_bin_size": 0,
        "line_var_y_bin_size": 0,
        "line_var_length": 0,
        "line_max_length": 0,
        "line_min_length": -1,
        "line_stroke_count": 0,
        "line_class_count": 0,
        "line_stroke_avg_bin_size": 0,
        "line_stroke_var_bin_size": 0,
    }
    svg_width = 960
    svg_height = 500
    svg_dimensions = get_svg_height_width(soup)
    if svg_dimensions:
        svg_width, svg_height = svg_dimensions
    svg_width = 1.0 * svg_width
    svg_height = 1.0 * svg_height
    soup, feature_dict = remove_axes(soup, feature_dict, svg_width, svg_height)
    paths = soup.find_all("path", {"d": lambda x: x is not None})
    path_count = len(paths)
    if path_count > 0:
        feature_dict = extract_path_features(paths, feature_dict, svg_width, svg_height)
    circles = soup.find_all("circle", {"r": lambda x: convert_str_to_float(x) > 0 if x is not None else False})
    circle_count = len(circles)
    if circle_count > 0:
        feature_dict = extract_circle_features(circles, feature_dict, svg_width, svg_height)
    rects = soup.find_all("rect")
    rect_count = len(rects)
    if rect_count > 0:
        feature_dict = extract_rect_features(rects, feature_dict, svg_width, svg_height)
    texts = soup.find_all("text")
    text_word_count = 0
    if len(texts) > 0:
        feature_dict, text_word_count = extract_text_features(texts, feature_dict, svg_width, svg_height)
    lines = soup.find_all("line")
    line_count = len(lines)
    if line_count > 0:
        feature_dict = extract_line_features(lines, feature_dict, svg_width, svg_height)
    all_shapes = 1.0 * (path_count + circle_count + rect_count + text_word_count + line_count)
    for key, count in [("path_count", path_count), ("circle_count", circle_count), ("rect_count", rect_count),
                       ("text_word_count", text_word_count), ("line_count", line_count)]:
        if all_shapes > 0:
            feature_dict[key] = count / all_shapes
        else:
            feature_dict[key] = 0
    if line_count > 0:
        feature_dict["circle_line_ratio"] = 1.0 * circle_count / line_count
    if path_count > 0:
        feature_dict["circle_path_ratio"] = 1.0 * circle_count / path_count
    squared_keys_width = ["path_x_variance", "circle_var_x", "rect_var_x", "rect_var_w", "line_var_x"]
    squared_keys_height = ["path_y_variance", "circle_var_y", "rect_var_y", "rect_var_h", "line_var_y"]
    squared_keys_max = ["circle_var_radius"]
    squared_keys_diagonal = ["line_var_length"]
    keys_max = ["circle_min_radius", "circle_max_radius"]
    keys_width = ["path_x_variance", "path_min_x_position", "path_max_x_position", "circle_max_x", "circle_min_x",
                  "rect_max_x", "rect_max_w", "rect_min_x", "rect_min_w", "line_max_x", "line_min_x"]
    keys_height = ["path_y_variance", "path_min_y_position", "path_max_y_position", "circle_max_y", "circle_min_y",
                   "rect_max_y", "rect_max_h", "rect_min_y", "rect_min_h", "line_max_y", "line_min_y"]
    keys_diagonal = ["circle_min_radius", "circle_max_radius"]
    dimensions = [svg_width, svg_height]
    max_dim = max(dimensions)
    diagonal = (svg_height ** 2 + svg_width ** 2) ** 0.5
    max_scale_factor = 1
    for keys, is_squared in [(squared_keys_width, True), (squared_keys_height, True), (squared_keys_max, True),
                             (squared_keys_diagonal, True),
                             (keys_width, False), (keys_height, False), (keys_max, False), (keys_diagonal, False)]:
        for key in keys:
            if feature_dict[key] > 1:
                if is_squared:
                    scale_factor = feature_dict[key] ** 0.5
                else:
                    scale_factor = feature_dict[key]
                if scale_factor > max_scale_factor:
                    max_scale_factor = scale_factor
    if max_scale_factor > 1:
        svg_width = max_scale_factor * svg_width
        svg_height = max_scale_factor * svg_height
        if feature_dict["path_count"] > 0:
            feature_dict = extract_path_features(paths, feature_dict, svg_width, svg_height)
        if feature_dict["circle_count"] > 0:
            feature_dict = extract_circle_features(circles, feature_dict, svg_width, svg_height)
        if feature_dict["rect_count"] > 0:
            feature_dict = extract_rect_features(rects, feature_dict, svg_width, svg_height)
        if feature_dict["line_count"] > 0:
            feature_dict = extract_line_features(lines, feature_dict, svg_width, svg_height)
    for key in list(feature_dict.keys()):
        if np.isnan(feature_dict[key]):
            feature_dict[key] = 0
        elif feature_dict[key] > sys.float_info.max:
            feature_dict[key] = 0
    #   for k in ['path_min_num_circular_arc_per_element', 'line_var_x_bin_size', 'path_min_stroke_width', 'path_y_variance', 'circle_min_x']:
    #  for k in ['circle_line_ratio', 'path_max_x_distance', 'text_unique_fill_count', 'text_min_font_size', 'rect_max_h', 'path_var_d_length_z', 'path_max_stroke_width', 'rect_unique_w_count', 'circle_max_radius', 'circle_count', 'circle_min_x', 'circle_max_x']:
    for k in ['circle_path_ratio', 'path_max_y_distance', 'circle_line_ratio', 'path_rx_var', 'line_avg_y_bin_size']:
        feature_dict.pop(k, None)
    return feature_dict
