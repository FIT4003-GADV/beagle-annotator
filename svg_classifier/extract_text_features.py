from utility import *
import numpy as np
import re
import sys

def extract_text_features(texts, feature_dict, svg_width, svg_height):
    fill_colors = set([])
    font_size_list = []
    word_count = 0
    x_set = set([])
    y_set = set([])
    for text in texts:
        content = text.getText()
        if convert_str_to_float(content) == 0:
            continue
        word_count += 1
        fill = None
        font_size = None
        if text.has_attr("style"):
            fill = get_value_from_style(text['style'], "fill")
            font_size = get_value_from_style(text['style'], "font-size")
        if fill:
            if is_colorful(fill):
                fill_colors.add(fill)
        if font_size:
            font_size = convert_str_to_float(font_size)
            font_size_list.append(font_size)
        tx, ty = get_location(text, 1, 0, 0)
        for loc, loc_set, t_var in [("x", x_set, tx), ("y", y_set, ty)]:
            loc_var = t_var
            if text.has_attr(loc):
                loc_var += convert_str_to_float(text[loc])
            loc_set.add(loc_var)
    if word_count == 0:
        return (feature_dict, word_count)
    if len(font_size_list) > 0:
        font_size_array = np.array(font_size_list)
        max_font_size = 1.0*np.max(font_size_array)
        feature_dict["text_max_font_size"] = 1.0
        feature_dict["text_min_font_size"] = np.min(font_size_array)/max_font_size
        feature_dict["text_var_font_size"] = np.var(font_size_array)/max_font_size**2
        feature_dict["text_unique_font_size_count"] = 1.0*len(set(font_size_list))/word_count
        feature_dict["text_unique_x_count"] = 1.0*len(x_set)/word_count
        feature_dict["text_unique_y_count"] = 1.0*len(y_set)/word_count
    feature_dict["text_unique_fill_count"] = 1.0*len(fill_colors)/word_count
    return (feature_dict, word_count)