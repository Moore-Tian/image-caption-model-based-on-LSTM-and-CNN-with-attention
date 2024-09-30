import fiftyone as fo
import fiftyone.zoo as foz

if __name__ == "__main__":  # Win10情况下必须把foz.load_zoo_dataset放到main下，否则多进程将会报错

    # 下载 2014 年的训练集
    train_dataset = foz.load_zoo_dataset(
        "coco-2014",  # 指定下载coco-2014类型
        split="train",  # 指定下载训练集
        label_types=["detections"],  # 指定下载目标检测的类型
        dataset_dir="COCO/",  # 指定下载的数据集保存的路径
        dataset_name="coco-2014-train",  # 指定新下载的数据集的名称
    )

    # 下载 2014 年的验证集
    val_dataset = foz.load_zoo_dataset(
        "coco-2014",  # 指定下载coco-2014类型
        split="validation",  # 指定下载验证集
        label_types=["detections"],  # 指定下载目标检测的类型
        dataset_dir="COCO/",  # 指定下载的数据集保存的路径
        dataset_name="coco-2014-val",  # 指定新下载的数据集的名称
    )

    # 启动 FiftyOne 应用查看训练集
    train_session = fo.launch_app(train_dataset)
    train_session.wait()

    # 启动 FiftyOne 应用查看验证集
    val_session = fo.launch_app(val_dataset)
    val_session.wait()
