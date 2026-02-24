document.addEventListener("DOMContentLoaded", () => {
    
    const AUTH_API = "http://localhost:5000"; 

    const accountIcon = document.getElementById("accountIcon");
    const accountDropdown = document.getElementById("accountDropdown");
    const loginEmail = document.getElementById("loginEmail");
    const loginPassword = document.getElementById("loginPassword");
    const registerFirstName = document.getElementById("registerFirstName");
    const registerLastName = document.getElementById("registerLastName");
    const registerEmail = document.getElementById("registerEmail");
    const registerPassword = document.getElementById("registerPassword");
    const userDisplay = document.getElementById("userDisplay");
    const authForms = document.getElementById("authForms");
    const registerForm = document.getElementById("registerForm");
    const logoutButton = document.getElementById("logoutButton");
    const subscriptionStatus = document.getElementById("subscriptionStatus");

    let currentUser = JSON.parse(sessionStorage.getItem("currentUser")) || null;

    /**
     * toggles the visibility of the account dropdown menu
     */
    accountIcon.addEventListener("click", (event) => {
        event.stopPropagation();
        accountDropdown.classList.toggle("show");
        updateAccountUI();
    });

    /**
     * closes the dropdown when clicking outside of it
     */
    document.addEventListener("click", (event) => {
        if (!accountDropdown.contains(event.target) && event.target !== accountIcon) {
            accountDropdown.classList.remove("show");
        }
    });

    /**
     * updates the UI based on whether a user is logged in or not
     */
    function updateAccountUI() {
        if (currentUser) {
            userDisplay.innerText = `Logged in as: ${currentUser.first_name} ${currentUser.last_name}`;
            authForms.style.display = "none";
            registerForm.style.display = "none";
            if (subscriptionStatus) {
                subscriptionStatus.innerText = `Subscription: ${currentUser.is_subscribed ? "Yes" : "No"}`;
                subscriptionStatus.style.display = "block";
            }
            logoutButton.style.display = "block";
        } else {
            userDisplay.innerText = "Account";
            authForms.style.display = "block";
            registerForm.style.display = "none";
            if (subscriptionStatus) {
                subscriptionStatus.style.display = "none";
            }
            logoutButton.style.display = "none";
        }
    }

    /**
     * fetches user session to persist login state
     */
    async function checkUserSession() {
        try {
            const response = await fetch(`${AUTH_API}/api/user`, {
                method: 'GET',
                credentials: 'include',
            });

            if (response.ok) {
                currentUser = await response.json();
                sessionStorage.setItem("currentUser", JSON.stringify(currentUser));
            } else {
                currentUser = null;
                sessionStorage.removeItem("currentUser");
            }
        } catch (error) {
            console.error("Error checking session:", error);
            currentUser = null;
            sessionStorage.removeItem("currentUser");
        }

        updateAccountUI();
    }

    /**
     * handles user login
     */
    window.handleLogin = async function () {
        const email = loginEmail.value.trim();
        const password = loginPassword.value.trim();

        if (!email || !password) {
            alert("Please fill in all fields.");
            return;
        }

        try {
            const response = await fetch(`${AUTH_API}/api/login`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await response.json();
            if (response.ok) {
                currentUser = data.user;
                sessionStorage.setItem("currentUser", JSON.stringify(currentUser));
                updateAccountUI();
                sessionStorage.setItem("currentUser", JSON.stringify(data.user));
                alert(data.message);
                window.location.reload(); 

            } else {
                alert(data.error);
            }
        } catch (error) {
            alert("An error occurred. Please try again.");
            console.error(error);
        }
    };

    /**
     * handles user registration
     */
    window.handleRegister = async function () {
        const firstName = registerFirstName.value.trim();
        const lastName = registerLastName.value.trim();
        const email = registerEmail.value.trim();
        const password = registerPassword.value.trim();

        if (!firstName || !lastName || !email || !password) {
            alert("Please fill in all fields.");
            return;
        }

        try {
            const response = await fetch(`${AUTH_API}/api/register`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ first_name: firstName, last_name: lastName, email, password }),
            });

            const data = await response.json();
            if (response.ok) {
                alert(data.message);
                registerForm.style.display = "none";
                authForms.style.display = "block";
            } else {
                alert(data.error);
            }
        } catch (error) {
            alert("An error occurred. Please try again.");
            console.error(error);
        }
    };

    /**
     * handles user logout
     */
    window.handleLogout = async function () {
        try {
            const response = await fetch(`${AUTH_API}/api/logout`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
            });

            let data;
            try {
                data = await response.json();
            } catch {
                data = { message: await response.text() };
            }

            if (!response.ok) {
                alert(data.error || "Logout failed.");
                return;
            }

            currentUser = null;
            sessionStorage.removeItem("currentUser");
            updateAccountUI();
            alert(data.message);
            window.location.reload(); 
        } catch (error) {
            alert("An error occurred. Please try again.");
            console.error(error);
        }
    };

    // switch to the register form
    document.getElementById("showRegister").addEventListener("click", () => {
        authForms.style.display = "none";
        registerForm.style.display = "block";
    });

    // switch to the login form
    document.getElementById("showLogin").addEventListener("click", () => {
        registerForm.style.display = "none";
        authForms.style.display = "block";
    });

    // attaches event listeners to each button
    document.getElementById("loginButton").addEventListener("click", handleLogin);
    document.getElementById("registerButton").addEventListener("click", handleRegister);
    document.getElementById("logoutButton").addEventListener("click", handleLogout);

    // check session on page load
    checkUserSession();

    
});
