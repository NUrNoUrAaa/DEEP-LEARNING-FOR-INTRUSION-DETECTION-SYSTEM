/* Netlify Function to proxy Flask API requests */
const http = require('http');

exports.handler = async (event, context) => {
  const path = event.path.replace('/.netlify/functions/api', '');
  const method = event.httpMethod;
  const headers = event.headers;
  const body = event.body;

  // CORS headers
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };

  // Handle preflight
  if (method === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: corsHeaders,
      body: '',
    };
  }

  try {
    const backendUrl = process.env.FLASK_API_URL || 'http://localhost:5000';
    
    const options = {
      hostname: new URL(backendUrl).hostname,
      port: new URL(backendUrl).port || 80,
      path: path,
      method: method,
      headers: {
        ...headers,
        'Host': new URL(backendUrl).hostname,
      },
    };

    return new Promise((resolve) => {
      const req = http.request(options, (res) => {
        let responseBody = '';

        res.on('data', (chunk) => {
          responseBody += chunk;
        });

        res.on('end', () => {
          resolve({
            statusCode: res.statusCode,
            headers: {
              ...corsHeaders,
              'Content-Type': res.headers['content-type'] || 'application/json',
            },
            body: responseBody,
          });
        });
      });

      req.on('error', (error) => {
        console.error('Backend request error:', error);
        resolve({
          statusCode: 502,
          headers: corsHeaders,
          body: JSON.stringify({ error: 'Backend service unavailable' }),
        });
      });

      if (body) {
        req.write(body);
      }
      req.end();
    });
  } catch (error) {
    console.error('Function error:', error);
    return {
      statusCode: 500,
      headers: corsHeaders,
      body: JSON.stringify({ error: 'Internal server error' }),
    };
  }
};
