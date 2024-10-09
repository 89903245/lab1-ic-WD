m =128;
n =256;
A = randn(m,n);
A = normc(A);
x = randn(n,1);

y= A*x;

x_1=pinv(A)*y;
x_2=A\y;

subplot(3,2,1)

plot(x)
title('oringinal x')
grid on;



subplot(3,2,2)
axis off;

subplot(3,2,3)
plot(x_1)
title('pinv(A)*y')
grid on;

subplot(3,2,4)
plot(x-x_1)
title('difference bettween original x and pinv(A)*y')
grid on;

subplot(3,2,5)

plot(x_2)
title('A\\y')
grid on;


subplot(3,2,6)
plot(x-x_2)
title('difference bettween original x and A\\y')
grid on;


