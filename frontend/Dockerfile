FROM nginx:alpine

# 移除預設設定
RUN rm /etc/nginx/conf.d/default.conf

# 放入我們的 nginx 設定
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 將 frontend 靜態檔案放進 nginx 預設目錄
COPY . /usr/share/nginx/html

EXPOSE 80
