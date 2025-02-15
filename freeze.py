import torch.nn as nn

class FreezeOtherBranches:
    """
    冻结指定网络分支以外的所有参数，使得这些参数不参与梯度计算。
    """
    def __init__(self, model, branch_name):
        """
        初始化函数，接受模型和要保留的分支名称。

        Args:
        model (nn.Module): 神经网络模型。
        branch_name (str): 要保留的模型分支的名称。
        """
        self.model = model
        self.branch_name = branch_name

    def freeze1(self):
        """
        冻结除了指定分支外的所有参数。
        """
        for name, param in self.model.named_parameters():
            if self.branch_name not in name:
                param.requires_grad = False

    def freeze2(self):
        """
        冻结指定分支的参数。
        """
        branch = getattr(self.model, self.branch_name)
        for param in branch.parameters():
            param.requires_grad = False

    def unfreeze1(self):
        """
        取消冻结除了指定分支外的所有参数，使得这些参数可以参与梯度计算。
        """
        for name, param in self.model.named_parameters():
            if self.branch_name not in name:
                param.requires_grad = True

    def unfreeze2(self):
        """
        取消冻结指定分支的参数，使得这些参数可以参与梯度计算。
        """
        branch = getattr(self.model, self.branch_name)
        for param in branch.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    # 示例模型
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.branch1 = nn.Sequential(
                nn.Linear(10, 50),
                nn.ReLU(),
                nn.Linear(50, 20)
            )
            self.branch2 = nn.Sequential(
                nn.Linear(20, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )

        def forward(self, x):
            x1 = self.branch1(x)
            x2 = self.branch2(x1)
            return x2

    # 示例使用
    model = MyModel()
    freezer = FreezeOtherBranches(model, 'branch1')

    # 冻结除 branch1 外的所有参数
    freezer.freeze()

    # 检查参数是否被冻结
    print("After freezing:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")  # branch1 的参数应为 True，其余应为 False

    # 取消冻结除 branch1 外的所有参数
    freezer.unfreeze()

    # 检查参数是否已经解冻
    print("\nAfter unfreezing:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")  # 所有参数应为 True
