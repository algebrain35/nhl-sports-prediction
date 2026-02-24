document.addEventListener("DOMContentLoaded", function () {
  let stripe, elements, clientSecret;

  const subscribeButton = document.getElementById("subscribeButton");
  const confirmButton = document.getElementById("confirmPaymentButton");
  const loader = document.getElementById("loader");
  const modal = document.getElementById("successModal");
  const paymentContainer = document.getElementById("payment-element-container");


  //unsubscribe button
  //  Check user subscription status on load
(async () => {
  let user = JSON.parse(sessionStorage.getItem("currentUser"));

  // If missing or not subscribed, fetch fresh data
  if (!user || user.is_subscribed === undefined) {
    try {
      const res = await fetch("http://localhost:5000/api/user", {
        method: "GET",
        credentials: "include"
      });

      const data = await res.json();
      if (data.email) {
        user = data;
        sessionStorage.setItem("currentUser", JSON.stringify(user));
      }
    } catch (err) {
      console.error("Error fetching user info:", err);
    }
  }

  if (user && user.is_subscribed) {
    subscribeButton.innerText = "Unsubscribe";
    subscribeButton.disabled = false;
    subscribeButton.style.opacity = "1";

    subscribeButton.addEventListener("click", async () => {
      const confirmed = confirm("Are you sure you want to cancel your subscription?");
      if (!confirmed) return;

      try {
        const res = await fetch("http://localhost:5000/api/stripe/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: user.email })
        });

        const data = await res.json();
        if (data.success) {
          alert("Subscription canceled.");
          sessionStorage.setItem("currentUser", JSON.stringify({ ...user, is_subscribed: false }));
          window.location.reload();
        } else {
          alert("Failed to cancel subscription.");
        }
      } catch (err) {
        alert("Error canceling subscription.");
        console.error(err);
      }
    });
    return; 
  }
})();



  const showLoader = () => loader.style.display = "flex";
  const hideLoader = () => loader.style.display = "none";

  // Subscribe Now
  subscribeButton.addEventListener("click", async (e) => {
    e.preventDefault();

    subscribeButton.disabled = true;


    const userData = JSON.parse(sessionStorage.getItem("currentUser")) || {};
    const email = userData.email || "test@example.com";
    const priceId = "price_1R6EshICYd8zgaTDu9zHa1a6";

    try {
      const res = await fetch("http://localhost:5000/api/stripe/subscription", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, price_id: priceId })
      });

      const data = await res.json();
      if (data.error) {
        alert("Error: " + data.error);
        subscribeButton.disabled = false;
        subscribeButton.innerText = "Subscribe Now";
        return;
      }

      clientSecret = data.clientSecret;
      stripe = Stripe(data.publishableKey);
      elements = stripe.elements({ clientSecret });

      const paymentElement = elements.create("payment");
      paymentElement.mount("#payment-element");

      subscribeButton.style.display = "none";

      paymentContainer.style.display = "block";
      paymentContainer.classList.add("slide-in");

    } catch (error) {
      console.error("Error creating subscription:", error);
      alert("An error occurred while creating the subscription.");
      subscribeButton.disabled = false;
      subscribeButton.innerText = "Subscribe Now";
    }
  });

  // Confirm Payment
  confirmButton.addEventListener("click", async (e) => {
    e.preventDefault();

    if (!stripe || !elements) {
      alert("Payment Element is not initialized yet.");
      return;
    }

    showLoader();

    const { error: submitError } = await elements.submit();
    if (submitError) {
      hideLoader();
      alert("Validation error: " + submitError.message);
      return;
    }

    const { error, paymentIntent } = await stripe.confirmPayment({
      elements,
      clientSecret,
      confirmParams: {
        return_url: "http://localhost:3000/predictions"
      },
      redirect: "if_required"
    });

    hideLoader();

    if (error) {
      console.error("Stripe confirmPayment error:", error);
      alert("Payment failed: " + error.message);
    } else if (paymentIntent?.status === "succeeded") {
      modal.classList.add("show");
      modal.style.display = "flex";
    //Refresh user session
      try {
        const res = await fetch("http://localhost:5000/api/user", {
          method: "GET",
          credentials: "include"
        });
        const updatedUser = await res.json();
        if (updatedUser.email) {
          sessionStorage.setItem("currentUser", JSON.stringify(updatedUser));
        }
      } catch (err) {
        console.error("Failed to refresh user session:", err);
      }

      setTimeout(() => {
        window.location.href = "predictions.html";
      }, 2000);
    } else {
      alert("Payment is processing. Please wait.");
    }
  });
});

