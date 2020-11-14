import re
import random

css_grey_color_names = {
"black":True,"darkgrey":True,"darkgray":True,"dimgrey":True,"dimgray":True,
"gainsboro":True,"gray":True,"grey":True,"lightgrey":True,"lightgray":True,"silver":True,"white":True
}
css_color_names = \
{"aliceblue":True,"antiquewhite":True,"aqua":True,"aquamarine":True,"azure":True,"beige":True,"bisque":True,"black":True,"blanchedalmond":True,"blue":True,"blueviolet":True,"brown":True,"burlywood":True,"cadetblue":True,"chartreuse":True,"chocolate":True,"coral":True,"cornflowerblue":True,"cornsilk":True,"crimson":True,"cyan":True,"darkblue":True,"darkcyan":True,"darkgoldenrod":True,"darkgray":True,"darkgrey":True,"darkgreen":True,"darkkhaki":True,"darkmagenta":True,"darkolivegreen":True,"darkorange":True,"darkorchid":True,"darkred":True,"darksalmon":True,"darkseagreen":True,"darkslateblue":True,"darkslategray":True,"darkslategrey":True,"darkturquoise":True,"darkviolet":True,"deeppink":True,"deepskyblue":True,"dimgray":True,"dimgrey":True,"dodgerblue":True,"firebrick":True,"floralwhite":True,"forestgreen":True,"fuchsia":True,"gainsboro":True,"ghostwhite":True,"gold":True,"goldenrod":True,"gray":True,"grey":True,"green":True,"greenyellow":True,"honeydew":True,"hotpink":True,"indianred":True,"indigo":True,"ivory":True,"khaki":True,"lavender":True,"lavenderblush":True,"lawngreen":True,"lemonchiffon":True,"lightblue":True,"lightcoral":True,"lightcyan":True,"lightgoldenrodyellow":True,"lightgray":True,"lightgrey":True,"lightgreen":True,"lightpink":True,"lightsalmon":True,"lightseagreen":True,"lightskyblue":True,"lightslategray":True,"lightslategrey":True,"lightsteelblue":True,"lightyellow":True,"lime":True,"limegreen":True,"linen":True,"magenta":True,"maroon":True,"mediumaquamarine":True,"mediumblue":True,"mediumorchid":True,"mediumpurple":True,"mediumseagreen":True,"mediumslateblue":True,"mediumspringgreen":True,"mediumturquoise":True,"mediumvioletred":True,"midnightblue":True,"mintcream":True,"mistyrose":True,"moccasin":True,"navajowhite":True,"navy":True,"oldlace":True,"olive":True,"olivedrab":True,"orange":True,"orangered":True,"orchid":True,"palegoldenrod":True,"palegreen":True,"paleturquoise":True,"palevioletred":True,"papayawhip":True,"peachpuff":True,"peru":True,"pink":True,"plum":True,"powderblue":True,"purple":True,"red":True,"rosybrown":True,"royalblue":True,"saddlebrown":True,"salmon":True,"sandybrown":True,"seagreen":True,"seashell":True,"sienna":True,"silver":True,"skyblue":True,"slateblue":True,"slategray":True,"slategrey":True,"snow":True,"springgreen":True,"steelblue":True,"tan":True,"teal":True,"thistle":True,"tomato":True,"turquoise":True,"violet":True,"wheat":True,"white":True,"whitesmoke":True,"yellow":True,"yellowgreen":True};

def convert_str_to_float(s):
    if "%" in s:
        return 0
    try:
        return float(s.rstrip("px").rstrip("em").rstrip("pt"))
    except Exception as e:
        return 0

def get_numbers_from_path(path):
    lower_path = path.lower()
    start = max(0, lower_path.rfind("m"))
    numbers = re.findall("([-\de\.]+)[\s,a-z]*", lower_path[start:])
    return [number for number in numbers if number != "e"]

# gets the value of the attribute specifed by keyword from the style attribute
def get_value_from_style(style, keyword):
    if style is None:
        return None
    re_keyword = re.compile(keyword + ":(.+?);")
    match = re_keyword.search(style)
    if match:
        return match.group(0)[len(keyword)+1:-1].replace(" ", "").replace("px", "").replace("em", "")
    return match

def find_attribute(element, attribute, include_parent, keywords):
    if include_parent:
        element_list = [element, element.parent]
    else:
        element_list = [element]
    for target in element_list:
        output = get_attribute(target, attribute)
        if output:
            return output
    if len(keywords) > 0:
        for target in element_list:
            if element.has_attr("class"):
                for keyword in keywords:
                    for e_class in element["class"]:
                        if keyword in e_class:
                            return e_class
    return None

def get_attribute(element, attribute):
    if element is None:
        return None
    if element.has_attr(attribute):
        return element[attribute]
    elif element.has_attr("style"):
        return get_value_from_style(element["style"], attribute)
    return None

# returns true if color is not None or the string "none"
def is_not_None(color):
    if color is None:
        return False
    if color.lower() == "none":
        return False
    return True

# returns true of color is None or is equal to the string "none"
def is_None(color):
    return (color is None) or (color.lower() == "none")

def color_specified(color):
    if is_None(color):
        return False
    if get_rgb_values(color) is not None:
        return True
    return color in css_color_names

# returns the list of numbers in the rgb(a) tuple
def get_rgb_values(color):
    if is_None(color):
        return None
    if color[0] == "#":
        return hex_to_rgb(color)
    match = re.compile("rgba?\(([^,]+),([^,]+),([^,]+)(?:,([^,]+))?\)").search(color)
    if match:
        num_groups = len(match.groups())
        return [int(match.group(i).replace(" ", "")) for i in range(1, num_groups)]
    return None

# converts the number in hex to rgb
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]

# determines if specified color is not a shade of grey
# does this for rgb by seeing if rbg values are not all equal
# does this for hex by converting to rgb
# does this for strings by looking for names that correspond to grey colors
def is_colorful(color):
    rgb_color = get_rgb_values(color)
    if rgb_color is None:
        return color not in css_grey_color_names
    else:
        if len(set(rgb_color)) == 1:
            return False
        return True


# returns a dictionary specifying the parameters for the specified arc_type (e.g. x1,y1,x,y for "q" arcs)
def get_arcs(path, arc_type):
    lower_path = path.lower()
    arc_indices = [m.start(0) for m in re.finditer(arc_type, lower_path)]
    if len(arc_indices) == 0:
        return []
    arcs = []
    if arc_type == "a":
        for i in range(len(arc_indices) - 1):
            arc_numbers = get_numbers_from_path(lower_path[arc_indices[i]+1:arc_indices[i+1]])
            if len(arc_numbers) < 7:
                continue
            arcs.append({'rx':convert_str_to_float(arc_numbers[0]), 'ry':convert_str_to_float(arc_numbers[1]), 'x':convert_str_to_float(arc_numbers[5]), 'y':convert_str_to_float(arc_numbers[6])})
        arc_numbers = get_numbers_from_path(lower_path[arc_indices[len(arc_indices) - 1]+1:])
        if len(arc_numbers) >= 7:
            arcs.append({'rx':convert_str_to_float(arc_numbers[0]), 'ry':convert_str_to_float(arc_numbers[1]), 'x':convert_str_to_float(arc_numbers[5]), 'y':convert_str_to_float(arc_numbers[6])})
    elif arc_type == "q":
        for i in range(len(arc_indices) - 1):
            arc_numbers = get_numbers_from_path(lower_path[arc_indices[i]+1:arc_indices[i+1]])
            if len(arc_numbers) != 4:
                continue
            arcs.append({'x1':convert_str_to_float(arc_numbers[0]), 'y1':convert_str_to_float(arc_numbers[1]), 'x':convert_str_to_float(arc_numbers[2]), 'y':convert_str_to_float(arc_numbers[3])})
        arc_numbers = get_numbers_from_path(lower_path[arc_indices[len(arc_indices) - 1]+1:])
        if len(arc_numbers) == 4:
            arcs.append({'x1':convert_str_to_float(arc_numbers[0]), 'y1':convert_str_to_float(arc_numbers[1]), 'x':convert_str_to_float(arc_numbers[2]), 'y':convert_str_to_float(arc_numbers[3])})
    return arcs

# gets the height and width of the svg element.  If there are multiple, gets the dimensions
# of the one with the largest area
def get_svg_height_width(soup):
    dimensions = []
    for svg in soup.find_all("svg"):
        if svg.has_attr("height") and svg.has_attr("width"):
            width = svg["width"]
            height = svg["height"]
            if height[-1] != "%" and width[-1] != "%":
                width = convert_str_to_float(width.replace("px", ""))
                height = convert_str_to_float(height.replace("px", ""))
                if width != 0 and height != 0:
                    dimensions.append((width, height))
    if len(dimensions) > 0:
        return max(dimensions, key=lambda x:x[0]*x[1])
    return None

# gets the location of the node as specified by the "transform" attribute.  If no location is found
# recurse to its parent
def get_location(node, count, x, y):
    if node is None or count > 3:
        return (x,y)
    if node.has_attr("transform"):
        transform = node["transform"].replace(" ", "").lower()
        match = re.compile("translate\(([^,a-z]+)(?:,([^a-z,]+))?\)").search(transform)
        if match:
            if match.group(2) is not None:
                x += convert_str_to_float(match.group(1))
                y += convert_str_to_float(match.group(2))
            else:
                y += convert_str_to_float(match.group(1))
    count += 1
    return get_location(node.parent, count, x, y)

# calculates the squared distance between two points
def distance2(x1,y1,x2,y2):
    return (x1 - x2)**2 + (y1 - y2)**2


# returns true if the line element is vertical or horizontal
def is_line_vertical_or_horizontal(line):
    attribute_dict = {"x1":0, "x2":0, "y1":0, "y2":0}
    for attribute in list(attribute_dict.keys()):
        if line.has_attr(attribute):
            if line[attribute][-1] != "%":
                attribute_dict[attribute] = convert_str_to_float(line[attribute])
    horizontal_difference = abs(attribute_dict["x1"] - attribute_dict["x2"])
    vertical_difference = abs(attribute_dict["y1"] - attribute_dict["y2"])
    if (vertical_difference == 0 and horizontal_difference > 0) or (horizontal_difference == 0 and vertical_difference > 0):
        return True
    return False

def reverse_dict(d):
    new_d = {}
    for (k, v) in d.items():
        new_d[v] = k
    return new_d