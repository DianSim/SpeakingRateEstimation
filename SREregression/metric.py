# implement pearcon cor coeff
import torch

def pearson_r(y_true, y_pred):
    # use smoothing for not resulting in NaN values
    epsilon = 10e-5
    x = y_true
    y = y_pred
    
    mx = x.mean()
    my = y.mean()
    print(mx)
    print(my)
    xm, ym = x - mx, y - my
    print(xm)
    print(ym)
    # r_num = K.sum(xm * ym)
    # x_square_sum = K.sum(xm * xm)
    # y_square_sum = K.sum(ym * ym)
    # r_den = K.sqrt(x_square_sum * y_square_sum)
    # r = r_num / (r_den + epsilon)
    return None #K.mean(r)


x = torch.Tensor([1, 2, 3, 4]).view(-1, 1)
y = torch.Tensor([2, 3, 4, 8]).view(-1, 1)
pearson_r(x, y)
