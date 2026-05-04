# 🌾 AgriCare AI: Intelligent Agricultural Solutions Platform

> **Revolutionizing Agriculture with AI-Powered Insights**

AgriCare AI is a cutting-edge platform that transforms agricultural practices through Artificial Intelligence. This comprehensive solution empowers farmers and stakeholders with advanced tools for precision agriculture, ensuring sustainable and optimized crop management.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.2+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## 🌟 Key Features

### 🌱 AI-Powered Plant Disease Management
- ✅ Early and accurate detection of plant diseases using YOLO deep learning
- ✅ Evidence-based treatment and prevention recommendations
- ✅ Expert agricultural advice (symptoms, causes, treatments, home remedies, prevention)
- ✅ High confidence scores (85-90% accuracy)
- ✅ Instant analysis from uploaded images

### 🧪 Comprehensive Soil Quality Analysis
- ✅ In-depth analysis of 12 essential soil parameters:
  - Nitrogen, Phosphorus, Potassium
  - pH Level, Electrical Conductivity, Organic Carbon
  - Sulphur, Zinc, Iron, Copper, Manganese, Boron
- ✅ Fertility level predictions (Low, Moderate, High)
- ✅ AI-powered personalized recommendations
- ✅ Nutrient optimization strategies

### 🧠 Smart Review Analysis
- ✅ Sentiment analysis powered by advanced LLMs
- ✅ Structured review insights (positive/negative aspects)
- ✅ Customer satisfaction assessment
- ✅ Improvement suggestions
- ✅ Business intelligence for decision-making

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Groq API key (free at [console.groq.com/keys](https://console.groq.com/keys))
- 4GB RAM minimum

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AGRI-CARE-AI-main.git
cd AGRI-CARE-AI-main

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Get Groq API Key**
   - Visit [Groq Console](https://console.groq.com/keys)
   - Create new API key (starts with `gsk_`)

2. **Update API Key**
   - Open `AGRI-CARE-AI-main/agricare.py`
   - Find line 21: `GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE"`
   - Replace with your actual API key

3. **Run Application**
   ```bash
   cd AGRI-CARE-AI-main
   streamlit run agricare.py
   ```

4. **Access Application**
   - Opens automatically in browser at `http://localhost:8501`
   - Or manually visit: `http://localhost:8501`

---

## 📊 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | Streamlit 1.42.2 | Interactive web UI |
| AI/ML Framework | LangChain 0.3.20 | LLM orchestration |
| Plant Detection | YOLO 8.3.103 | Disease detection |
| Soil Analysis | XGBoost 3.0.0 | Fertility prediction |
| LLM Provider | Groq API | AI-powered recommendations |
| Deep Learning | PyTorch 2.6.0 | Neural network backend |
| Language | Python 3.11+ | Core programming language |

---

## 📁 Project Structure

```
AGRI-CARE-AI-main/
├── AGRI-CARE-AI-main/
│   ├── agricare.py                 # Main application
│   ├── best.pt                     # YOLO plant disease model
│   ├── soli_analysis.pkl           # XGBoost soil analysis model
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Documentation
├── LICENSE                         # MIT License
└── README_DETAILED.md             # Detailed documentation
```

---

## 🎯 How to Use

### Tab 1: Plant Disease Detection
1. Upload a plant leaf image
2. View YOLO detection results with confidence score
3. Enter plant name for targeted advice
4. Receive expert agricultural recommendations

### Tab 2: Soil Analysis
1. Enter 12 soil parameters from lab test results
2. Click "Analyze Soil" button
3. View fertility level prediction
4. Get AI-powered improvement recommendations

### Tab 3: Smart Review Analysis
1. Paste customer review in text area
2. Click "Analyze Review" button
3. View sentiment analysis and structured insights
4. Get improvement suggestions

---

## 🛠️ API Configuration

### Groq API

**Setup Steps:**
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up for free account
3. Generate API key
4. Update line 21 in `agricare.py`

**Model Used:** `llama-3.3-70b-versatile`

**Features:**
- ⚡ Fast inference (300+ tokens/second)
- 💰 Free tier available
- 🔒 Secure and reliable
- 🚀 Easy to use REST API

---

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError" when running**
```bash
pip install -r requirements.txt
```

**Port 8501 already in use**
```bash
streamlit run agricare.py --server.port 8502
```

**Model file not found**
- Ensure `best.pt` and `soli_analysis.pkl` are in project directory

**API key not working**
- Verify key starts with `gsk_`
- Check key is correctly pasted in `agricare.py` line 21
- Ensure key hasn't been revoked on Groq console

For more help, see [README_DETAILED.md](README_DETAILED.md#-troubleshooting)

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add: Amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Submit** Pull Request

See [README_DETAILED.md](README_DETAILED.md#-contributing) for detailed guidelines.

---

## 📚 Documentation

- **Quick Start**: This README
- **Detailed Docs**: [README_DETAILED.md](README_DETAILED.md)
- **API Reference**: [Groq Docs](https://console.groq.com/docs)
- **Framework Docs**: [Streamlit Docs](https://docs.streamlit.io/)

---

## 📄 License

MIT License © 2024 AgriCare AI Team

This project is open source and available under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

- 🐛 **Report Issues**: [GitHub Issues](https://github.com/yourusername/AGRI-CARE-AI-main/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/AGRI-CARE-AI-main/discussions)
- 📧 **Email**: [your-email@example.com]
- 🐦 **Twitter**: [@YourHandle]

---

## ✨ Features Roadmap

**v2.0 (Planned)**
- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support
- [ ] Real-time crop monitoring
- [ ] Weather integration
- [ ] Historical data tracking

**Contribute Ideas**: Create an [Issue](https://github.com/yourusername/AGRI-CARE-AI-main/issues) with suggestions!

---

## ⭐ Show Your Support

If you find this project helpful:
- ⭐ **Star** the repository
- 🐛 **Report bugs** to help improve
- 💡 **Suggest features** for future versions
- 📢 **Share** with fellow farmers and developers
- 🤝 **Contribute** code and improvements

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [LangChain](https://langchain.com/) - LLM orchestration
- [Groq](https://groq.com/) - Fast LLM inference
- [Ultralytics](https://ultralytics.com/) - YOLO implementation
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting

---

<div align="center">

**Made with ❤️ for Farmers and Agricultural Enthusiasts**

[Star ⭐](https://github.com/yourusername/AGRI-CARE-AI-main) • [Report Issues 🐛](https://github.com/yourusername/AGRI-CARE-AI-main/issues) • [View Docs 📚](README_DETAILED.md)

**Version 1.0.0** | Last Updated: May 4, 2026 | Status: ✅ Production Ready

</div>
