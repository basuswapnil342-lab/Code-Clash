# Code-Clash
Innovating new ideas to improve khatabook and make it a better company from before by our ideas
# Khatabook AI-Powered Smart Insights Base Code
# This is a basic Python implementation for a digital ledger app inspired by Khatabook.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import os

class KhatabookLedger:
    def __init__(self, data_file='transactions.csv'):
        self.data_file = data_file
        if os.path.exists(self.data_file):
            self.transactions = pd.read_csv(self.data_file)
            self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        else:
            self.transactions = pd.DataFrame(columns=['date', 'customer', 'type', 'amount', 'description'])
            self.save_data()

    def add_transaction(self, customer, transaction_type, amount, description=''):
        """Add a new transaction to the ledger.
        - transaction_type: 'credit' or 'debit'
        """
        new_transaction = {
            'date': datetime.now(),
            'customer': customer,
            'type': transaction_type,
            'amount': amount,
            'description': description
        }
        self.transactions = pd.concat([self.transactions, pd.DataFrame([new_transaction])], ignore_index=True)
        self.save_data()

    def save_data(self):
        """Save transactions to CSV file."""
        self.transactions.to_csv(self.data_file, index=False)

    def get_balance(self, customer=None):
        """Get balance for a specific customer or total."""
        if customer:
            df = self.transactions[self.transactions['customer'] == customer]
        else:
            df = self.transactions
        credits = df[df['type'] == 'credit']['amount'].sum()
        debits = df[df['type'] == 'debit']['amount'].sum()
        return credits - debits

    def get_insights(self):
        """Provide basic insights."""
        print("\n--- Basic Insights ---")
        total_balance = self.get_balance()
        print(f"Total Balance: {total_balance}")

        customer_balances = self.transactions.groupby('customer').apply(lambda x: x[x['type'] == 'credit']['amount'].sum() - x[x['type'] == 'debit']['amount'].sum())
        print("\nCustomer Balances:")
        print(customer_balances)

        monthly_summary = self.transactions.resample('M', on='date')['amount'].sum()
        print("\nMonthly Transaction Summary:")
        print(monthly_summary)

    def ai_predict_future_balance(self, days_ahead=30):
        """AI-Powered Insight: Predict future balance using linear regression."""
        if len(self.transactions) < 2:
            print("Not enough data for prediction.")
            return

        # Prepare data for prediction
        self.transactions['day'] = (self.transactions['date'] - self.transactions['date'].min()).dt.days
        daily_balance = self.transactions.groupby('day').apply(lambda x: x[x['type'] == 'credit']['amount'].sum() - x[x['type'] == 'debit']['amount'].sum()).cumsum()
        daily_balance = daily_balance.reset_index(name='cumulative_balance')

        # Linear Regression Model
        model = LinearRegression()
        X = daily_balance['day'].values.reshape(-1, 1)
        y = daily_balance['cumulative_balance'].values
        model.fit(X, y)

        # Predict future
        future_days = np.array([daily_balance['day'].max() + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)
        predictions = model.predict(future_days)

        print("\n--- AI-Powered Future Balance Prediction ---")
        for day, pred in zip(range(1, days_ahead + 1), predictions):
            print(f"Day {day}: Predicted Balance = {pred:.2f}")

        # Plot
        plt.plot(daily_balance['day'], daily_balance['cumulative_balance'], label='Historical Balance')
        plt.plot(future_days, predictions, label='Predicted Balance', linestyle='--')
        plt.xlabel('Days')
        plt.ylabel('Balance')
        plt.title('Balance Prediction')
        plt.legend()
        plt.show()

# Example Usage
if __name__ == "__main__":
    ledger = KhatabookLedger()

    # Add sample transactions
    ledger.add_transaction('Customer1', 'credit', 1000, 'Payment received')
    ledger.add_transaction('Customer1', 'debit', 500, 'Goods supplied')
    ledger.add_transaction('Customer2', 'credit', 2000, 'Advance payment')

    # Get insights
    ledger.get_insights()

    # AI Prediction
    ledger.ai_predict_future_balance(7)

