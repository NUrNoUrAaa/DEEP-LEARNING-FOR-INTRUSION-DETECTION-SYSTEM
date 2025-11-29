# Deploying to Netlify

## Steps to Deploy on Netlify:

### 1. Connect Repository
1. Go to https://app.netlify.com/
2. Click "New site from Git"
3. Select GitHub and authorize
4. Choose this repository: `DEEP-LEARNING-FOR-INTRUSION-DETECTION-SYSTEM`

### 2. Build Settings (Auto-detected)
The `netlify.toml` file will handle these automatically:
- **Build Command**: `npm run build`
- **Publish Directory**: `dist/ids-dashboard/browser`

### 3. Environment Variables (Optional)
If needed, add environment variables in:
Site settings → Build & deploy → Environment

### 4. Deployment
After connecting:
- Every push to `main` branch will trigger automatic deployment
- Netlify will run the build command
- Angular app will be deployed to your Netlify domain

### 5. Custom Domain (Optional)
- Site settings → Domain management
- Add your custom domain (if you have one)

---

## Deploy Status
View deployment status at: https://app.netlify.com/

## Notes
- GitHub Pages deployment is disabled
- Using Netlify for production deployment only
- Automatic deployments on every push to main branch
