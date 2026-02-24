document.addEventListener("DOMContentLoaded", async () => {
    const AUTH_API = "http://localhost:5000/api/user";

    const teamSelect = document.getElementById("team-select");
    const historyTableHead = document.querySelector("#history-table thead");
    const historyTableBody = document.querySelector("#history-table tbody");
    const getHistoryBtn = document.getElementById("get-history");

    if (!teamSelect) {
        console.error("Team dropdown not found!");
        return;
    }

    /**
     * fills the dropdown menus with team abbreviations
     * each dropdown starts with a default "select team" option
     * adds all nhl teams as options
     */
    function fillTeamDropdown() {
        const teams = [
            "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
            "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
            "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG"
        ];

        teamSelect.innerHTML = '<option value="">Select Team</option>';
        teams.forEach(team => {
            teamSelect.add(new Option(team, team));
        });
    }

    /**
     * fetches team history from the backend and renders it as a table
     * splits large data into chunks and renders them asynchronously
     */
    async function fetchTeamHistory() {
        const team = teamSelect.value;

        if (!team) {
            alert("Please select a team.");
            return;
        }

        try {
            const response = await fetch(`http://localhost:5000/api/nhl/teams/data?team=${team}`);
            const data = await response.json();

            historyTableHead.innerHTML = "";
            historyTableBody.innerHTML = "";

            const records = Array.isArray(data) ? data : Object.values(data);
            const chunkSize = 10;
            let index = 0;

            if (records.length === 0) {
                historyTableBody.innerHTML = "<tr><td colspan='100%'>No history found.</td></tr>";
                return;
            }

            /**
             * creates table headers dynamically from the first record
             */
            const keys = Object.keys(records[0]);
            const headerRow = document.createElement("tr");
            keys.forEach(key => {
                const th = document.createElement("th");
                th.textContent = key;
                headerRow.appendChild(th);
            });
            historyTableHead.appendChild(headerRow);

            /**
             * renders rows in chunks for performance
             */
            function renderChunk() {
                const chunk = records.slice(index, index + chunkSize);
                chunk.forEach(record => {
                    const row = document.createElement("tr");
                    keys.forEach(key => {
                        const td = document.createElement("td");
                        td.textContent = record[key];
                        row.appendChild(td);
                    });
                    historyTableBody.appendChild(row);
                });

                index += chunkSize;
                if (index < records.length) {
                    setTimeout(renderChunk, 100);
                }
            }

            renderChunk();
        } catch (error) {
            historyTableBody.innerHTML = "<tr><td colspan='100%'>Error fetching history.</td></tr>";
            console.error(error);
        }
    }

    /**
     * checks if the user is subscribed before allowing access
     * redirects to pricing page if not subscribed
     */
    async function checkSubscription() {
        try {
            const response = await fetch(AUTH_API, { credentials: "include" });
            if (!response.ok) throw new Error("Failed to fetch user data.");

            const user = await response.json();

            if (!user.is_subscribed) {
                window.location.href = "pricing.html";
                return;
            }

            sessionStorage.setItem("currentUser", JSON.stringify(user));
        } catch (error) {
            console.error("error checking subscription:", error);
            window.location.href = "pricing.html";
        }
    }

    getHistoryBtn.addEventListener("click", fetchTeamHistory);
    await checkSubscription();
    fillTeamDropdown();
});
