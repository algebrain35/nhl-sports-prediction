import stripe
import os
import logging
from dotenv import load_dotenv
from models import User, db

load_dotenv() 

logging.getLogger("stripe").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def create_subscription(email, price_id):
    customer = stripe.Customer.create(email=email)

    subscription = stripe.Subscription.create(
        customer=customer.id,
        items=[{"price": price_id}],
        payment_behavior="default_incomplete",
        expand=["latest_invoice", "latest_invoice.payment_intent"],
    )

    payment_intent = subscription.latest_invoice.payment_intent


    logger.info("Subscription created", extra={
        "subscription_id": subscription.id,
        "customer_id": subscription.customer,
        "payment_intent_id": payment_intent.id,
        "status": payment_intent.status
    })


    return {
        "subscriptionId": subscription.id,
        "clientSecret": subscription.latest_invoice.payment_intent.client_secret,
        "publishableKey": os.getenv("STRIPE_PUBLISHABLE_KEY")
    }

def cancel_subscription(email):
    user = User.query.filter_by(email=email).first()
    if not user:
        return False, "User not found"

    try:
        subscriptions = stripe.Subscription.list(limit=10)

        for sub in subscriptions.auto_paging_iter():
            customer = stripe.Customer.retrieve(sub.customer)
            if customer.email == email and sub.status == "active":
                stripe.Subscription.delete(sub.id)

                user.is_subscribed = False
                db.session.commit()
                return True, None

        return False, "No active subscription found"
    except Exception as e:
        return False, str(e)