# USAGE
## Step 1. Clone this github
```cmd
git clone https://github.com/k19tvan/AIC_LOCAL.git
cd AIC_LOCAL
```
## Step 2. 
### If keyframes were downloaded
- Move webp_keyframes folder into AIC_LOCAL
### Otherwise
- Download data from server.
    - Window
    
        ```cmd
        scp -r nguyenmv@192.168.20.156:/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Retrieval/Keyframes/webp_keyframes ./
        ```
    - Linux
        ```cmd
        rsync -aP nguyenmv@192.168.20.156:/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Retrieval/Keyframes/webp_keyframes ./
        ```

## Step 3. Install Docker
https://docs.docker.com/engine/install/

## Step 4. Host front end 
```cmd
docker run --name host_web -p 2108:80 -v ./:/usr/share/nginx/html:ro -d nginx
docker start host_web 
```

The website will be http://localhost:2108 on browser.