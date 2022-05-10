from mypackge.ch05 import MulLayer, AddLayer

if __name__ == '__main__':
    apple = 100
    organe=150
    apple_num = 2
    organe_num=3
    tax = 1.1

    #layer
    mul_apple_layer=MulLayer()
    mul_organe_layer = MulLayer()
    add_apple_organe_layer=AddLayer()
    mul_tax_layer=MulLayer()

    #forward

    apple_price=mul_apple_layer.forward(apple,apple_num)
    organe_price = mul_organe_layer.forward(organe, organe_num)
    print("apple_price = ",apple_price,"organe_price = ",organe_price)
    all_price=add_apple_organe_layer.forward(apple_price,organe_price)
    print("all_price = ", all_price)
    price=mul_tax_layer.forward(all_price,tax)
    print("price = ", price)
    print('*'*15)

    # backward
    dprice = 1
    dall_price,dtax=mul_tax_layer.backward(dprice)
    print("dall_price = ", dall_price, "dtax = ", dtax)

    dapple_price,dorgane_price=add_apple_organe_layer.backward(dall_price)
    print("dapple_price = ", dapple_price, "dorgane_price = ", dorgane_price)

    dapple,dapple_num=mul_apple_layer.backward(dapple_price)
    print("dapple = ", dapple, "dapple_num = ", dapple_num)

    dorgane, dorgane_num = mul_organe_layer.backward(dorgane_price)
    print("dorgane = ", dorgane, "dorgane_num = ", dorgane_num)
