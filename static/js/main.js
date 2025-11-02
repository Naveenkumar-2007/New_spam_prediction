document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('messageForm');
    const messageInput = document.getElementById('messageInput');
    const clearButton = document.getElementById('clearButton');
    const formMessage = document.getElementById('formMessage');

    const resultsSection = document.getElementById('resultsSection');
    const predictionTag = document.getElementById('predictionTag');
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const messagePreview = document.getElementById('messagePreview');

    const setFormMessage = (text, tone = 'info') => {
        formMessage.textContent = text;
        formMessage.dataset.tone = tone;
        formMessage.hidden = !text;
        if (text) {
            formMessage.className = `form-message ${tone}`;
        }
    };

    const toggleResults = (visible) => {
        resultsSection.hidden = !visible;
    };

    const updateResults = ({ prediction, confidence, message }) => {
        const isSpam = prediction === 'Spam';
        predictionTag.textContent = isSpam ? 'Spam detected' : 'Legitimate';
        predictionTag.className = `tag ${isSpam ? 'spam' : 'safe'}`;
        predictionText.textContent = isSpam
            ? 'This message looks suspicious. Review carefully before taking action.'
            : 'This message appears legitimate. No obvious spam signals detected.';
        confidenceValue.textContent = `${confidence.toFixed(2)}%`;
        messagePreview.textContent = message.length > 160
            ? `${message.substring(0, 157)}…`
            : message;
        toggleResults(true);
    };

    const handleError = (message) => {
        setFormMessage(message, 'error');
        toggleResults(false);
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const message = messageInput.value.trim();

        if (!message) {
            handleError('Please enter a message before running the prediction.');
            return;
        }

        setFormMessage('Analyzing message...', 'info');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                const errorText = data.message || 'Unable to score the message right now.';
                handleError(errorText);
                return;
            }

            setFormMessage('Prediction complete.', 'success');
            updateResults({
                prediction: data.prediction,
                confidence: Number(data.confidence) || 0,
                message: data.message || message
            });
        } catch (error) {
            console.error('Prediction request failed', error);
            handleError('Network error while contacting the prediction service.');
        }
    });

    clearButton.addEventListener('click', () => {
        messageInput.value = '';
        setFormMessage('');
        toggleResults(false);
        messageInput.focus();
    });
});
