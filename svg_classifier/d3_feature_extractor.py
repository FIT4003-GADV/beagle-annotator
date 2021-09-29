import sys

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
    add_rect = False
    soup = BeautifulSoup(open(filename, 'r').read(),"html.parser")
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
    "path_max_displacement": 0,
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
    "path_stroke_avg_bin_size": 0,
    "path_stroke_var_bin_size": 0,
    "path_fill_avg_bin_size": 0,
    "path_fill_var_bin_size": 0,
    "circle_line_ratio": 0,
    "circle_path_ratio": 0,
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
    "circle_avg_fill_bin_size": 0,
    "circle_var_fill_bin_size": 0,
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
    "rect_avg_fill_bin_size": 0,
    "rect_var_fill_bin_size": 0,
    "text_word_count": 0,
    "text_max_font_size": 0,
    "text_min_font_size": -1,
    "text_var_font_size": 0,
    "text_unique_font_size_count": 0,
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
    "line_stroke_avg_bin_size": 0,
    "line_stroke_var_bin_size": 0,
    }
    svg_width = 960
    svg_height = 500
    svg_dimensions = get_svg_height_width(soup)
    if svg_dimensions:
        svg_width, svg_height = svg_dimensions
    svg_width = 1.0*svg_width
    svg_height = 1.0*svg_height
    soup, feature_dict = remove_axes(soup, feature_dict, svg_width, svg_height)
    paths = soup.find_all("path", {"d":lambda x: x is not None})
    path_count = len(paths)
    if path_count > 0:
        feature_dict = extract_path_features(paths, feature_dict, svg_width, svg_height)
    circles = soup.find_all("circle", {"r":lambda x: convert_str_to_float(x) > 0 if x is not None else False})
    circle_count = len(circles)
    if circle_count > 0:
        feature_dict = extract_circle_features(circles, feature_dict, svg_width, svg_height)
    rects = soup.find_all("rect")
    rect_count = len(rects)
    if rect_count > 0:
        feature_dict = extract_rect_features(rects, feature_dict, svg_width, svg_height)
    else:
        add_rect = True
    texts = soup.find_all("text")
    text_word_count = 0
    if len(texts) > 0:
        feature_dict, text_word_count = extract_text_features(texts, feature_dict, svg_width, svg_height)
    lines = soup.find_all("line")
    line_count = len(lines)
    if line_count > 0:
        feature_dict = extract_line_features(lines, feature_dict, svg_width, svg_height)
    all_shapes = 1.0*(path_count + circle_count + rect_count + text_word_count + line_count)
    for key, count in [("path_count", path_count), ("circle_count", circle_count), ("rect_count", rect_count), ("text_word_count", text_word_count), ("line_count", line_count)]:
        if all_shapes > 0:
            feature_dict[key] = count/all_shapes
        else:
            feature_dict[key] = 0
    if line_count > 0:
        feature_dict["circle_line_ratio"] = 1.0*circle_count/line_count
    if path_count > 0:
        feature_dict["circle_path_ratio"] = 1.0*circle_count/path_count
    squared_keys_width = ["path_x_variance", "circle_var_x", "rect_var_x", "rect_var_w", "line_var_x"]
    squared_keys_height = ["path_y_variance", "circle_var_y", "rect_var_y", "rect_var_h", "line_var_y"]
    squared_keys_max = ["circle_var_radius"]
    squared_keys_diagonal = ["line_var_length"]
    keys_max = ["circle_min_radius", "circle_max_radius"]
    keys_width = ["path_x_variance", "path_min_x_position", "path_max_x_position", "circle_max_x", "circle_min_x", "rect_max_x", "rect_max_w", "rect_min_x", "rect_min_w", "line_max_x", "line_min_x"]
    keys_height = ["path_y_variance", "path_min_y_position", "path_max_y_position", "circle_max_y", "circle_min_y", "rect_max_y", "rect_max_h", "rect_min_y", "rect_min_h", "line_max_y", "line_min_y"]
    keys_diagonal = ["circle_min_radius", "circle_max_radius"]
    dimensions = [svg_width, svg_height]
    max_dim = max(dimensions)
    diagonal = (svg_height**2 + svg_width**2)**0.5
    max_scale_factor = 1
    for keys, is_squared in [(squared_keys_width, True), (squared_keys_height, True), (squared_keys_max, True), (squared_keys_diagonal, True),
        (keys_width, False), (keys_height, False), (keys_max, False), (keys_diagonal, False)]:
        for key in keys:
            if feature_dict[key] > 1:
                if is_squared:
                    scale_factor = feature_dict[key]**0.5
                else:
                    scale_factor = feature_dict[key]
                if scale_factor > max_scale_factor:
                    max_scale_factor = scale_factor
    if max_scale_factor > 1:
        svg_width = max_scale_factor*svg_width
        svg_height = max_scale_factor*svg_height
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
    # for k in [
    #     "path_max_x_distance",
    #     "rect_var_h",
    #     "line_stroke_var_bin_size",
    #     "rect_max_h",
    #     "path_max_y_distance",
    #     "rect_max_w",
    #     "rect_var_h_bin_size",
    #     "rect_avg_h_bin_size",
    #     "path_num_noncircular_arcs",
    #     "path_avg_num_noncircular_arc_per_element",
    #     "rect_var_w_bin_size",
    #     "rect_var_w",
    #     "path_max_num_noncircular_arc_per_element",
    #     "text_min_font_size",
    #     "rect_max_w_count",
    #     "line_stroke_count",
    #     "rect_min_h",
    #     "rect_avg_w_bin_size",
    #     "path_stroke_var_bin_size",
    #     "text_max_font_size",
    #     "rect_min_w",
    #     "path_min_num_noncircular_arc_per_element"]:
    for k in [
        "path_max_x_distance",
        "rect_var_h",
        "rect_max_h",
        "path_max_y_distance",
        "rect_max_w",
        "rect_var_h_bin_size",
        "path_num_noncircular_arcs",
        "path_avg_num_noncircular_arc_per_element",
       "rect_var_w_bin_size",
       "rect_var_w",
        "path_max_num_noncircular_arc_per_element"
        # "text_word_count",
        # "text_max_font_size",
        # "text_min_font_size",
        # "text_var_font_size",
        # "text_unique_font_size_count",
        # "text_unique_x_count",
        # "text_unique_y_count"
        ]:
        feature_dict.pop(k, None)
    if add_rect:
        feature_dict["colorful_rect_count"] = 0.0
    return feature_dict









    # vertices, edges = get_vertices_edges(soup)
    # feature_dict["has_vertices_and_edges"] = int(len(vertices) > 0)
    # sankey_paths = get_sankey_paths(soup)
    # feature_dict["sankey_paths"] = int(has_sankey_paths(sankey_paths))
    # boxes = get_boxes(soup)
    # feature_dict["boxes"] = int(len(boxes) > 0)
    # if feature_dict["radial_axes"] == 0:
    #     feature_dict["radial_axes"] = int(has_radial_axes(soup))
    # hexagons = get_hexagons(soup)
    # feature_dict["hexagons"] = int(len(hexagons) > 5)
    # slice_indicator, donut_slices = get_donut_or_pie_slices(soup)
    # if slice_indicator == 0:
    #     feature_dict["stacked_donut_slices"] = int(has_two_pairs_of_stacked_donut_slices(donut_slices))
    # feature_dict["treemap_blocks"] = int(has_treemap_blocks(soup))
    # if feature_dict["stacked_donut_slices"] == 0 and slice_indicator == 1:
    #     feature_dict["donut_slices"] = int(len(donut_slices) > 0)
    # if slice_indicator == 2:
    #     feature_dict["pie_slices"] = int(len(donut_slices) > 0)
    # chords = get_chords(soup)
    # feature_dict["chords"] = int(len(chords) > 0)
    # word_cloud_words = get_word_cloud_words(soup, svg_width, svg_height)
    # feature_dict["word_cloud_words"] = int(len(word_cloud_words) > 0)
    # is_rectangular, waffle_blocks = get_waffle_blocks(soup)
    # heatmap_blocks = get_heatmap_blocks(soup)
    # if len(waffle_blocks) > 0 and len(heatmap_blocks) > 0 and is_rectangular:
    #     feature_dict["num_heatmap_waffle_blocks"] = max(len(heatmap_blocks), len(waffle_blocks))
    # if not is_rectangular:
    #     feature_dict["waffle_blocks"] = int(len(waffle_blocks) > 0)
    # # feature_dict["waffle_blocks"] = int(len(waffle_blocks) > 7)
    # # if feature_dict["waffle_blocks"] == 0:
    # #     heatmap_blocks = get_heatmap_blocks(soup)
    # #     feature_dict["heatmap_blocks"] = int(len(heatmap_blocks) > 0)
    # if feature_dict["hexagons"] == 0:
    #     feature_dict["voronoi_blocks"] = get_voronoi_blocks(soup, svg_width, svg_height)
    # if feature_dict["voronoi_blocks"] == 0:
    #     feature_dict["has_map_features"] = int(len(get_map_features(soup)) > 0)
    # if feature_dict["has_vertices_and_edges"] == 0 and feature_dict["radial_axes"] == 0 and feature_dict["voronoi_blocks"] == 0:
    #     all_circles = [x[0] for x in get_filled_circles(soup)]
    #     filled_circle_group = get_group_with_lowest_common_ancestor(soup, all_circles)
    #     filled_circles = []
    #     if filled_circle_group:
    #         filled_circles = get_filled_circles(filled_circle_group)
    #     feature_dict["filled_circles"] = int(len(filled_circles) > 1)
    #     if feature_dict["filled_circles"] == 1:
    #         max_radius = max(filled_circles, key=lambda x:x[1])[1]
    #         min_radius = min(filled_circles, key=lambda x:x[1])[1]
    #         feature_dict["max_min_filled_circle_radii_ratio"] = max_radius/min_radius

    # if feature_dict["num_heatmap_waffle_blocks"] == 0 and feature_dict["treemap_blocks"] == 0:
    #     # get the filled circles features
    #     feature_dict.update(get_filled_circles_features(soup))
    # if feature_dict["num_heatmap_waffle_blocks"] == 0 and not feature_dict["treemap_blocks"]:
    #     bars = get_bars(soup)
    #     # checks to see if there are two rect elements with same width but different
    #     # heights and vice versa
    #     widths = {}
    #     for bar in bars:
    #         if bar['width'] not in widths.keys():
    #             widths[bar['width']] = set([bar['height']])
    #         else:
    #             widths[bar['width']].add(bar['height'])
    #             if len(widths[bar['width']]) > 1:
    #                 feature_dict["bars"] = 1
    #                 break
    #     if feature_dict["bars"] == 0:
    #         heights = {}
    #         for bar_h in bars:
    #             if bar_h['height'] not in heights.keys():
    #                 heights[bar_h['height']] = set([bar_h['width']])
    #             else:
    #                 heights[bar_h['height']].add(bar_h['width'])
    #                 if len(heights[bar_h['height']]) > 1:
    #                     feature_dict["bars"] = 1
    #                     break
    # if feature_dict["hexagons"] == 0 and feature_dict["has_map_features"] == 0 and feature_dict["donut_slices"] == 0 and feature_dict["voronoi_blocks"] == 0:
    #     areas, avg_vertical_area_distance_from_center = get_areas(soup, svg_height)
    #     feature_dict["areas"] = int(len(areas) > 0)
    #     if feature_dict["areas"] == 1:
    #         feature_dict["avg_vertical_area_distance_from_center"] = avg_vertical_area_distance_from_center
    # if feature_dict["has_vertices_and_edges"] == 0 and feature_dict["sankey_paths"] == 0 and feature_dict["chords"] == 0 and feature_dict["boxes"] == 0 and feature_dict["has_map_features"] == 0 and feature_dict["voronoi_blocks"] == 0:
    #     lines = get_lines(soup, svg_width)
    #     feature_dict["lines"] = int(len(lines) > 0)

    # return feature_dict









#print len(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1282/svg.txt"))

#features = extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1167/svg.txt")

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/268/svg.txt")) + " graph"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1414/svg.txt")) + " graph"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1091/svg.txt")) + " line"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1758/svg.txt")) + " map"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1433/svg.txt")) + " chord"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1033/svg.txt")) + " bubble"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/131/svg.txt")) + " parallel coordinates"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1237/svg.txt")) + " parallel coordinates"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/638/svg.txt")) + " sankey"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/751/svg.txt")) + " box"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/164/svg.txt")) + " area"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/188/svg.txt")) + " stream_graph"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/493/svg.txt")) + " heatmap"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/426/svg.txt")) + " radial"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1271/svg.txt")) + " hexabin"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1394/svg.txt")) + " sunburst"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/2305/svg.txt")) + " treemap"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/734/svg.txt")) + " voronoi"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1431/svg.txt")) + " donut"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1369/svg.txt")) + " pie"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/3005/svg.txt")) + " word_cloud"

#print str(extract("../d3_chart_sample_complete_svg/3012/svg.txt")) + " waffle"

#print str(extract("/Users/duanp/meng_project/webTablesScraper/d3classification/d3_chart_sample_complete_svg/1545/svg.txt")) + " test"

# for k in features.keys():
#     if "line" in k or "rect" in k:
#         print k
#        print features[k]
