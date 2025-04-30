const express = require("express");

const cors = require("cors");

const uploadRoute = require("./upload");

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));


// Routes
app.use("/", uploadRoute);



// Start the server
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
