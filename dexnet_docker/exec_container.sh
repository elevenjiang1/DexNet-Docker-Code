# can set shortcut in ~/.bashrc
# alias dexnet='sudo docker exec -it dexnet bash'
sudo docker start dexnet
sudo xhost +local:`docker inspect --format='{{ .Config.Hostname }}' dexnet`
docker exec -it dexnet bash
