const express = require('express'); 
const path = require('path'); 
const cors = require('cors'); 
const http = require('http'); 
const app = express(); 

const PORT = 3000; 

// enables cors so the frontend can make requests to the API
app.use(cors());

// serves the html pages
app.use(express.static(path.join(__dirname, '..')));

// endpoint for the moneyline predictions API (proxy between frontend and the API)
app.get('/api/nhl/ml/predict', (req, res) => {
    // extract the home and away team parameters in the qu
    const { home, away } = req.query; 
    
    // error if no team is provided
    if (!home || !away) {
        return res.status(400).json({ error: "Missing 'home' or 'away' team parameter" });
    }

    // api request url and fills the team parameters
    const predictionsAPI = `http://localhost:5000/api/nhl/ml/predict?home=${home}&away=${away}`;

    // sends the request to the API
    http.get(predictionsAPI, (flaskRes) => {
        let data = '';

        // waits for the data from the API
        flaskRes.on('data', (chunk) => {
            data += chunk; // appends the api to a variable
        });

        // when the response is recieved, ...
        flaskRes.on('end', () => {
            try {
                // parse the data into JSON
                const jsonResponse = JSON.parse(data);
                // sends the JSON to the frontend
                res.json(jsonResponse);
            } catch (error) {
                res.status(500).json({ error: "Invalid JSON from API" });
            }
        });
    }).on('error', (err) => {
        console.error("Error fetching predictions from API:", err.message);
        res.status(500).json({ error: "Error fetching predictions" });
    });
});

// gets the endpoint for the OU predictions API (proxy between frontend and the API)
app.get('/api/nhl/ou/predict', (req, res) => {
    let { home, away, ou } = req.query;

    // error if the parameter is missing
    if (!home || !away || !ou) {
        return res.status(400).json({ error: "Missing 'home', 'away', or 'ou' parameter" });
    }

    // converts the OU value to a float
    ou = parseFloat(ou);

    // error if the OU value is not a valid number
    if (isNaN(ou) || ou <= 0) {
        return res.status(400).json({ error: "Invalid 'ou' parameter. Must be a positive number." });
    }

    // formats the OU value to one decimal
    ou = ou.toFixed(1);

    // url for the API request
    const predictionsAPI = `http://localhost:5000/api/nhl/ou/predict?home=${home}&away=${away}&ou=${ou}`;

    // sends the request to the API
    http.get(predictionsAPI, (flaskRes) => {
        let data = '';

        // waits for the data from the API
        flaskRes.on('data', (chunk) => {
            data += chunk;
        });

        // when the response is recieved, ...
        flaskRes.on('end', () => {
            try {
                // parse the data into JSON
                const jsonResponse = JSON.parse(data);
                res.json(jsonResponse);
            } catch (error) {
                res.status(500).json({ error: "Invalid JSON from API" });
            }
        });
    }).on('error', (err) => {
        console.error("Error fetching OU predictions from API:", err.message);
        res.status(500).json({ error: "Error fetching OU predictions" });
    });
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
