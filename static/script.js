document.addEventListener('DOMContentLoaded', () => {
    const navbar = document.getElementById('navbar');
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const riskBadge = document.getElementById('risk-badge');
    const probFill = document.getElementById('prob-fill');
    const probText = document.getElementById('prob-text');
    const riskMsg = document.getElementById('risk-msg');

    // Navbar Scroll Effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        const submitBtn = form.querySelector('.predict-btn');
        const originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Analyzing Geo-Data...';

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.status === 'success') {
                // Update UI
                resultDiv.classList.remove('hidden');
                
                const probPercent = (result.probability * 100).toFixed(1);
                probText.textContent = `${probPercent}%`;
                probFill.style.width = `${probPercent}%`;

                riskBadge.textContent = result.risk_level;
                riskBadge.className = 'badge'; // reset
                
                // Color mapping
                if (result.risk_level === 'Very Low' || result.risk_level === 'Low') {
                    riskBadge.classList.add('badge-low');
                    riskMsg.textContent = "The area appears to be stable. However, always remain vigilant during heavy monsoon seasons.";
                } else if (result.risk_level === 'Moderate') {
                    riskBadge.classList.add('badge-moderate');
                    riskMsg.textContent = "Moderate risk detected. Periodic monitoring of terrain and drainage systems is recommended.";
                } else {
                    riskBadge.classList.add('badge-high');
                    riskMsg.textContent = "CAUTION: Significant landslide risk identified. Consider engineering interventions and follow local drainage clearance protocols.";
                }

                // Scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('An error occurred while processing your request.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;
        }
    });

    // Add CSS for spinner dynamically
    const style = document.createElement('style');
    style.innerHTML = `
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
});
