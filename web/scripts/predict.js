document.addEventListener("DOMContentLoaded", async () => {
    const AUTH_API = "http://localhost:5000/api/user"; // api endpoint for user data

    const homeSelect = document.getElementById("home-team");
    const awaySelect = document.getElementById("away-team");
    const ouInput = document.getElementById("ou-value");
    const spreadInput = document.getElementById("spread-value"); 
    const predictionResult = document.getElementById("ml-result");
    const ouResult = document.getElementById("ou-result");
    const spreadResult = document.getElementById("spread-result"); 
    const getPredictionBtn = document.getElementById("get-prediction");
    const getOUPredictionBtn = document.getElementById("get-ou-prediction");
    const getSpreadPredictionBtn = document.getElementById("get-spread-prediction"); 

    if (!homeSelect || !awaySelect) {
        console.error("dropdown elements not found!");
        return;
    }

    /**
     * fills the dropdown menus with team abbreviations
     * each dropdown starts with a default "select team" option
     * adds all nhl teams as options
     */
    function fillTeamDropdowns() {
        const teams = [
            "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
            "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
            "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG"
        ];

        homeSelect.innerHTML = '<option value="">select home team</option>';
        awaySelect.innerHTML = '<option value="">select away team</option>';

        teams.forEach(team => {
            homeSelect.add(new Option(team, team));
            awaySelect.add(new Option(team, team));
        });
    }

    /**
     * checks if the user is subscribed
     * fetches user data from the backend api
     * if the user is not subscribed, redirects them to the pricing page
     * saves user data in session storage for future use
     */
    async function checkSubscription() {
        try {
            const response = await fetch(AUTH_API, { credentials: "include" }); // sends request with credentials
            if (!response.ok) throw new Error("failed to fetch user data.");

            const user = await response.json(); // parses the json data

            if (!user.is_subscribed) {
                window.location.href = "pricing.html"; // redirects to pricing page if not subscribed
                return;
            }

            sessionStorage.setItem("currentUser", JSON.stringify(user)); // stores user data in session storage
        } catch (error) {
            console.error("error checking subscription:", error);
            window.location.href = "pricing.html"; // redirects if error occurs
        }
    }

    /**
     * fetches the moneyline prediction based on selected teams
     * sends request to the backend api with the selected teams as query parameters
     * displays the win probabilities for both teams
     */
    async function fetchPrediction() {
        const home = homeSelect.value;
        const away = awaySelect.value;

        if (!home || !away) {
            predictionResult.textContent = "please select both teams.";
            return;
        }

        try {
            const response = await fetch(`/api/nhl/ml/predict?home=${home}&away=${away}`);
            const data = await response.json();
            const [homeWinProb, awayWinProb] = data.predictions[0];

            predictionResult.innerHTML = `
                <p><strong>${home} Win probability:</strong> ${(awayWinProb * 100).toFixed(2)}%</p>
                <p><strong>${away} Win probability:</strong> ${(homeWinProb * 100).toFixed(2)}%</p>
            `;
        } catch (error) {
            predictionResult.textContent = "error fetching ml prediction.";
            console.error(error);
        }
    }

    /**
     * fetches the over/under prediction based on selected teams and the user-entered value
     * sends request to the backend api with teams and over/under value as query parameters
     * displays the probability of over and under outcomes
     */
    async function fetchOUPrediction() {
        const home = homeSelect.value;
        const away = awaySelect.value;
        const ouValue = ouInput.value;

        if (!home || !away || !ouValue) {
            ouResult.textContent = "please select both teams and enter an over/under value.";
            return;
        }

        try {
            const response = await fetch(`/api/nhl/ou/predict?home=${home}&away=${away}&ou=${ouValue}`);
            const data = await response.json();
            const [underProb, overProb] = data.predictions[0];

            ouResult.innerHTML = `
                <p><strong> Over ${ouValue} probability:</strong> ${(overProb * 100).toFixed(1)}%</p>
                <p><strong> Under ${ouValue} probability:</strong> ${(underProb * 100).toFixed(1)}%</p>
            `;
        } catch (error) {
            ouResult.textContent = "error fetching ou prediction.";
            console.error(error);
        }
    }

    /**
     * fetches the spread prediction based on selected teams and the user entered value
     * sends request to the backend api with teams and spread value as query parameters
     * displays the probability of spread outcomes
     */

    async function fetchSpreadPrediction() {
        const home = homeSelect.value;
        const away = awaySelect.value;
        const spreadValue = spreadInput.value;
    
        if (!home || !away || !spreadValue) {
            spreadResult.textContent = "please select both teams and enter a spread value.";
            return;
        }
    
        try {
            const response = await fetch(`http://localhost:5000/api/nhl/spread/predict?home=${home}&away=${away}&spread=${spreadValue}`);
            const data = await response.json();
            const [homeWinProb, awayWinProb] = data.predictions[0];
    
            spreadResult.innerHTML = `
                <p><strong> Probability of ${home} covering spread ${spreadValue} is:</strong> ${(awayWinProb* 100).toFixed(2)}%</p>
                <p><strong> Probability of ${away} covering spread ${-spreadValue} is :</strong> ${(homeWinProb * 100).toFixed(2)}%</p>
            `;
        } catch (error) {
            spreadResult.textContent = "error fetching spread prediction.";
            console.error(error);
        }
    }
    

    // adds event listeners to the buttons for fetching predictions
    getPredictionBtn.addEventListener("click", fetchPrediction);
    getOUPredictionBtn.addEventListener("click", fetchOUPrediction);
    getSpreadPredictionBtn.addEventListener("click", fetchSpreadPrediction); // added event listener for spread

    // fills dropdown menus first, then checks subscription status
    fillTeamDropdowns();
    await checkSubscription();
});
