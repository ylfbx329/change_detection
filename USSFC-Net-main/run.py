from main import main
import argparse


def get_params():
    """
    获取命令行参数

    :return: args
    """
    parser = argparse.ArgumentParser(description='RSCD_PyTorch')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    # 返回少量参数，多余参数不报错
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    # 返回字典
    params = vars(get_params())
    main(params)
