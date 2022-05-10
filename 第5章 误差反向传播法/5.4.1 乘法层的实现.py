from mypackge.ch05 import MulLayer

if __name__ == '__main__':
    apple = 100
    apple_num = 2
    tax = 1.1
    # layer
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()
    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print("apple_price = ",apple_price,"price = ",price)
    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    print("dapple_price = ",dapple_price,"dtax = ",dtax)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print("dapple = ", dapple, "dapple_num = ", dapple_num)

