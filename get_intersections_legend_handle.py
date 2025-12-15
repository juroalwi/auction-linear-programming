# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.patches import Patch


# def get_intersections_legend_handle(grids, names, colors, alpha_fill):
#     K = len(grids)
#     N, M = grids[0].shape

#     codes = np.zeros((N, M), dtype=np.uint8)
#     for i, mask in enumerate(grids):
#         codes |= mask.astype(np.uint8) << i

#     unique_codes = np.unique(codes)
#     unique_codes = unique_codes[unique_codes > 0]

#     handles = []

#     for c in unique_codes:
#         idxs = [i for i in range(K) if (c >> i) & 1]
#         label = " + ".join(names[i] for i in idxs)
#         colors_rgb = np.array([mcolors.to_rgb(colors[i]) for i in idxs])
#         blended_rgb = np.asarray(
#             mcolors.to_rgb(plt.gca().get_facecolor()), dtype=float
#         )  # start with background color
#         for color_rgb in colors_rgb:  # blend with each color
#             blended_rgb = color_rgb * alpha_fill + blended_rgb * (1.0 - alpha_fill)

#         handles.append(Patch(facecolor=blended_rgb, edgecolor="black", label=label))

#     return handles
