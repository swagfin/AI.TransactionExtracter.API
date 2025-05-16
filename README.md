# AI.TransactionExtracter.API

A lightweight AI-powered .NET API that extracts key entities (e.g., **amounts**, **reference numbers**) from transactional messages like SMS or email alerts.

Built using [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet), this solution provides minimalistic AI capabilities suitable for use in Docker containers and constrained environments.

---

## 🛠️ Features

- ✅ Extracts **Amount** and **Reference Number** from free-form transaction messages (More to be added soon)
- ✅ Fast and lightweight model using `SdcaMaximumEntropy`.
- ✅ Fully .NET-based: no Python dependencies.
- ✅ Easily train and update your own model using `training-data.tsv.tsv` files.

---

## 🔍 Example Input
```txt
"You have received 1500 KES in your account. Ref 987654."
```

### 🧠 Predicted Output
```sh
AMOUNT: 1500
REFERENCENUMBER: 987654
```
