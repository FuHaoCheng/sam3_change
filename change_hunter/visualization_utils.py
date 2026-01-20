import matplotlib.pyplot as plt
from sam3.visualization_utils import COLORS, plot_mask, plot_bbox


def plot_results_save(img, results, path):
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")

    ax = plt.gca()

    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]

        mask_t = results["masks"][i].squeeze(0).cpu()   # HxW
        plot_mask(mask_t, color=color)

        # ✅ 白色边框（轮廓线）
        mask_np = (mask_t > 0.5).numpy().astype(float)
        ax.contour(mask_np, levels=[0.5], colors="white", linewidths=2)  # 线宽可调

        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            # text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )

    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()