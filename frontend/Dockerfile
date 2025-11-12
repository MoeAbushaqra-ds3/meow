FROM node:18

WORKDIR /app

COPY package.json package-lock.json ./

ENV NODE_PATH=/app/node_modules
ENV PATH=$PATH:/app/node_modules/.bin

RUN npm ci

COPY . .

CMD ["npm", "run", "dev"]
