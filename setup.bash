echo 123456 | sudo cp ./99-fixed-interbotix-udev.rules /etc/udev/rules.d/99-fixed-interbotix-udev.rules
echo 123456 | sudo udevadm control --reload
echo 123456 | sudo udevadm trigger
ls /dev | grep ttyDXL
