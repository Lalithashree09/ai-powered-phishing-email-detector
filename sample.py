# Save as create_sample.py
import pandas as pd

data = {
    'sender': ['alert@bank.com', 'support@paypal.com', 'friend@gmail.com'],
    'receiver': ['user@domain.com']*3,
    'date': ['2025-01-01']*3,
    'subject': ['Urgent: Verify Account', 'Payment Failed', 'Hi, how are you?'],
    'body': [
        'Your account will be locked. Click http://fakebank.com/login to verify.',
        'We detected unusual activity. Login at http://paypal-secure.co to fix.',
        'Just checking in. Hope you are doing well!'
    ],
    'label': ['phishing', 'phishing', 'ham'],
    'urls': [
        '["http://fakebank.com/login"]',
        '["http://paypal-secure.co"]',
        '[]'
    ]
}

df = pd.DataFrame(data)
df.to_csv('TREC_07.csv', index=False)
print("Sample TREC_07.csv created!")