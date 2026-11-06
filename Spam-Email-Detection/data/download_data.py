"""
data/download_data.py
---------------------
Utility script to download or prepare the SMS Spam Collection dataset.
Source: UCI Machine Learning Repository / Kaggle SMS Spam Collection.
"""

import os
import urllib.request
import pandas as pd

# Define relative path to dataset
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "spam.csv")

# Reliable mirror of the official SMS Spam Collection dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)


def download_dataset():
    """
    Downloads the SMS Spam Collection dataset and saves it as data/spam.csv
    with standard columns ['label', 'message'].
    """
    if os.path.exists(DATA_FILE):
        print(f"[INFO] Dataset already exists at {DATA_FILE}")
        return

    print(f"[INFO] Downloading dataset from {DATASET_URL}...")
    try:
        # Download the tab-delimited file
        temp_tsv = os.path.join(DATA_DIR, "temp_sms.tsv")
        urllib.request.urlretrieve(DATASET_URL, temp_tsv)

        # Read TSV with columns 'label' and 'message'
        df = pd.read_csv(temp_tsv, sep="\t", header=None, names=["label", "message"])

        # Save as standard CSV
        df.to_csv(DATA_FILE, index=False)

        # Clean up temporary TSV
        if os.path.exists(temp_tsv):
            os.remove(temp_tsv)

        print(f"[SUCCESS] Dataset successfully saved to {DATA_FILE}")
        print(f"[INFO] Total records downloaded: {len(df)}")
        print(df.head())
    except Exception as e:
        print(f"[WARNING] Could not download dataset online: {e}")
        print("[INFO] Creating a local bootstrap dataset for offline operation...")
        create_bootstrap_dataset()


def create_bootstrap_dataset():
    """
    Creates an initial bootstrap dataset with representative spam and ham messages
    in case internet connection is unavailable.
    """
    sample_data = {
        "label": [
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"
        ],
        "message": [
            "Hey, are we still meeting for lunch today at 12:30?",
            "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card. Call 0800123456 now to claim your prize! T&C apply.",
            "Can you please review the attached document before our team call this afternoon?",
            "URGENT! Your bank account has been compromised. Verify your login details immediately at http://secure-bank-login.com",
            "I'll be running a few minutes late to the sync meeting, please start without me.",
            "WINNER!! As a valued customer you are selected to receive a free £900 reward! Text CLAIM to 87121 now.",
            "Thanks for sending the lecture notes over, they were really helpful for the exam prep.",
            "FREE entry into our weekly £1000 prize draw. Text WON to 85023 to enter. 18+ only.",
            "Don't forget Mom's birthday dinner this Saturday evening at 7 PM.",
            "Exclusive offer: Get 80% discount on luxury watches! Limited time only. Click here: www.discount-watches.fake",
            "Are you free this weekend to help me move some boxes into the new apartment?",
            "Final notice: You have an unclaimed parcel pending delivery. Pay $1.99 redelivery fee at http://parcel-tracker-scam.net",
            "The project presentation is ready. Let's do a quick rehearsal at 3 PM.",
            "You have won a free iPhone 15 Pro! Claim your prize within the next 24 hours by registering at http://free-phone-now.org",
            "Hi, did you get the chance to check out the GitHub repository I shared yesterday?",
            "Loan approved! Instant cash transfer up to $50,000 with 0% interest rate for the first 6 months. Apply now!",
            "I will pick up the groceries on my way home from work.",
            "Alert! Your credit card has suspicious activity. Reply YES to confirm or click http://credit-verify-update.com",
            "Could you send me the Zoom link for today's discussion on machine learning?",
            "Get rich quick with automated crypto trading! Guaranteed 200% return in 48 hours. Sign up today at http://crypto-scam.io",
            "Let me know when you arrive at the airport so I can pull up to terminal 2.",
            "Hot singles in your area want to chat with you right now! Click here to connect.",
            "Please find the meeting minutes attached for yesterday's sprint planning.",
            "You have been awarded 500 bonus points! Redeem them for gift vouchers now at http://loyalty-reward-prize.com",
            "What time does the library close on Friday evening?",
            "STOP! You have 1 unread message waiting from customer support. Call 09061743813 to listen to your voice message.",
            "See you tomorrow morning at the coffee shop.",
            "Earn $500 per day working from home with no experience needed! Start immediately at http://easy-money-fast.biz",
            "Can you email me the updated budget spreadsheet when you get a chance?",
            "Dear customer, your Netflix subscription has expired. Update your billing method at http://netflix-billing-update.fake",
            "I finished reviewing your pull request, left a few minor comments on GitHub.",
            "Ring tones! Download the top 10 hit ringtones for free. Reply TONE to 80088 to activate service.",
            "Great job on the presentation today, everyone was really impressed by the results.",
            "Congratulations! Your mobile number won $500,000 in the International Lottery. Send your details to claim@lottery.org",
            "Are we having class tomorrow or was it rescheduled for next week?",
            "Cash prize alert! Call 09061701461 to claim your £5000 prize. Costs £1.50 per min.",
            "Let's grab some coffee and discuss the assignment questions after lunch.",
            "Act now! Special pharmaceutical discount on prescription medicine. No prescription needed: www.cheap-meds.fake",
            "Did you hear back from the recruiter regarding the interview schedule?",
            "Your PayPal account has been temporarily restricted. Log in to verify your identity: http://paypal-security-alert.net",
            "I will be working remotely tomorrow due to internet maintenance in the building.",
            "Double your Bitcoin today! Send 0.1 BTC to receive 0.2 BTC instantly. Offer ends tonight.",
            "Don't worry about cooking tonight, I already ordered takeout for both of us.",
            "FREE gift card giveaway! Complete our 1-minute survey to receive a $50 gift card immediately.",
            "Can you water my plants while I'm away this weekend? I left the spare key under the mat.",
            "Selected user! You have been chosen to test the new iPad and keep it for free. Visit http://tester-rewards.net",
            "Happy birthday! Wishing you a wonderful year ahead filled with happiness and success.",
            "Urgent notification: Tax refund of $842 is ready to be deposited. Submit bank details at http://tax-refund-gov.fake",
            "The professor posted the sample solutions for problem set 4 on the portal.",
            "Risk-free trial! Lose 15 pounds in 10 days with our miracle weight loss pills. Order today!",
            "I will send the code review feedback by end of day.",
            "Exclusive VIP invitation! Free spins and $200 bonus chips at our online casino. Play now!",
            "Let me know if you want to carpool to the conference on Thursday morning.",
            "Congratulations! You won 2 tickets to the Premier League final. Reply TICKET to 84025 to claim.",
            "Looking forward to catching up with everyone at the alumni reunion.",
            "Security code alert: Someone attempted to access your Amazon account. Verify your account immediately at http://amazon-auth-check.net",
            "Please review the slide deck before tomorrow's client presentation.",
            "Instant cash loan! No credit check required. Cash deposited in 15 minutes. Call 0800-LOAN-NOW.",
            "I'm heading to the gym now, let me know if you want to join.",
            "You have 1 new voicemail from unknown caller. Click here to listen: http://voicemail-notification.fake",
            "Thanks for helping out with the data analysis yesterday, saved me a lot of time.",
            "Claim your $250 Amazon voucher today! Click the link below to verify your email address.",
            "Is the team lunch still scheduled for Thursday at noon?",
            "URGENT: Your parcel delivery is pending payment of customs duty. Clear fees here: http://parcel-customs-fee.com"
        ]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(DATA_FILE, index=False)
    print(f"[SUCCESS] Created bootstrap dataset with {len(df)} samples at {DATA_FILE}")


if __name__ == "__main__":
    download_dataset()

