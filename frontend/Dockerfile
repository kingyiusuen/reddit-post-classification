# Use the latest node image
FROM node:latest

# Create a directory called app
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copies all the files in the frontend directory
COPY . .

# Run the app
CMD npm start