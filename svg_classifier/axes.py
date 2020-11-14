from utility import *

# Identifies and removes axes from chart.  D3 axes are straight path elements with two tick marks at the end
# and their lengths are at least half that of the svg element.  This function also looks for axes that are
# arranged radially, which is indicative of radial charts

def remove_axes(soup, feature_dict, svg_width, svg_height):
    paths = soup.find_all("path", {"d":lambda x: x is not None})
    vertical_axes = []
    horizontal_axes = []
    for path in paths:
        d = path["d"].lower()
        num_v = len([m.start() for m in re.finditer('v', d)])
        num_h = len([m.start() for m in re.finditer('h', d)])
        if num_h == 1 and num_v <= 2 and 'z' not in d:
            if len(get_numbers_from_path(d)) < 7 and get_axes_length(d, False) > 0.5*svg_width:
                horizontal_axes.append(path)
                path.extract()
        if num_v == 1 and num_h <= 2 and 'z' not in d:
            if len(get_numbers_from_path(d)) < 7 and get_axes_length(d, True) > 0.5*svg_height:
                vertical_axes.append(path)
                path.extract()
    feature_dict["horizontal_axes_count"] = len(horizontal_axes)
    feature_dict["vertical_axes_count"] = len(vertical_axes)
    return (soup, feature_dict)


# gets the length of the axis elemebt, which may be vertical of horizontal

def get_axes_length(path, is_vertical):
    if is_vertical:
        v_index = path.find('v')
        y_start = get_numbers_from_path(path[:v_index])[-1]
        y_end = get_numbers_from_path(path[v_index:])[0]
        return abs(convert_str_to_float(y_end) - convert_str_to_float(y_start))
    else:
        h_index = path.find('h')
        x_start = get_numbers_from_path(path[:h_index])[-2]
        x_end = get_numbers_from_path(path[h_index:])[0]
        return abs(convert_str_to_float(x_end) - convert_str_to_float(x_start))