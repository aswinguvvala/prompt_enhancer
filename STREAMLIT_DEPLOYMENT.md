# Streamlit Deployment Guide

This guide explains how to deploy the AI Prompt Enhancement Studio on Streamlit Cloud.

## Quick Start (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   Or use the helper script:
   ```bash
   python run_streamlit.py
   ```

## Streamlit Cloud Deployment

### Step 1: Prepare Repository

1. **Fork or clone this repository** to your GitHub account
2. **Ensure these files are present:**
   - `streamlit_app.py` (main app)
   - `requirements.txt` (dependencies)
   - `backend/` folder with all necessary modules

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Click "New app"**
3. **Connect your GitHub repository**
4. **Set the following:**
   - **Repository:** your-username/prompt-enhancement-system
   - **Branch:** main
   - **Main file path:** streamlit_app.py

### Step 3: Configure Secrets

In Streamlit Cloud, go to your app settings and add secrets:

```toml
[secrets]
OPENAI_API_KEY = "your_openai_api_key_here"
```

### Step 4: Deploy

Click "Deploy" and wait for the app to build and start.

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration

The app uses GPT-4o Mini by default for cost-effective prompt enhancement. This is configured in `backend/config.py`:

```python
OPENAI_MODEL: str = "gpt-4o-mini"  # Cost-effective model
```

## Features

- **Model-Specific Enhancement:** Optimizes prompts for Claude, GPT-4, Gemini, and Grok
- **Real-time Enhancement:** Uses GPT-4o Mini for fast, cost-effective processing
- **Professional UI:** Clean, responsive Streamlit interface
- **Before/After Comparison:** Shows original vs enhanced prompts side-by-side

## Cost Estimation

Using GPT-4o Mini:
- **Input cost:** ~$0.15 per 1M tokens
- **Output cost:** ~$0.60 per 1M tokens
- **Typical enhancement:** ~500 input + 200 output tokens = ~$0.0002 per enhancement
- **With $5 budget:** ~25,000 prompt enhancements

## Troubleshooting

### Common Issues

1. **"OpenAI API Key Required" error:**
   - Ensure `OPENAI_API_KEY` is set in Streamlit secrets
   - Verify the API key is valid and has credits

2. **Module import errors:**
   - Check that all files are in the repository
   - Ensure `backend/` folder structure is preserved

3. **Deployment fails:**
   - Check `requirements.txt` for correct dependencies
   - Verify Python version compatibility (3.8+)

### Testing Locally

Run the test script to verify everything works:

```bash
python test_openai_integration.py
```

This will test:
- OpenAI API connectivity
- Model rules loading
- Complete enhancement pipeline

## Architecture

```
streamlit_app.py          # Main Streamlit application
├── backend/
│   ├── config.py         # Configuration settings
│   ├── simplified_guides.py  # Model-specific rules
│   └── models/
│       └── prompt_enhancer.py  # AI model integration
├── requirements.txt      # Dependencies
├── run_streamlit.py     # Local runner script
└── test_openai_integration.py  # Test script
```

## Security Notes

- Never commit API keys to the repository
- Use Streamlit secrets for sensitive configuration
- API keys are only used for OpenAI API calls
- No data is stored or logged permanently

## Support

If you encounter issues:

1. Check the [Streamlit documentation](https://docs.streamlit.io/)
2. Verify your OpenAI API key and credits
3. Test locally before deploying to cloud
4. Check the app logs in Streamlit Cloud dashboard