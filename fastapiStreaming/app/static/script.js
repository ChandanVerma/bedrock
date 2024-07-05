async function generateFeedback() {
    const feedback = document.getElementById("feedback").value;
  
    if (feedback.trim().length === 0) {
      return;
    }
  
    const storyOutput = document.getElementById("story-output");
    storyOutput.innerText = "Thinking...";
  
    try {
      // Use Fetch API to send a POST request for response streaming. See https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API 
      const response = await fetch("/api/write_with_ai", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ "feedback": feedback })
      });
  
      storyOutput.innerText = "";
  
      // Response Body is a ReadableStream. See https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
  
      // Process the chunks from the stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const text = decoder.decode(value);
        storyOutput.innerText += text;
      }
  
    } catch (error) {
      storyOutput.innerText = `Sorry, an error happened. Please try again later. \n\n ${error}`;
    }
  
  }
  
  document.getElementById("get-feedback").addEventListener("click", generateFeedback);
  document.getElementById('feedback').addEventListener('keydown', function (e) {
    if (e.code === 'Enter') {
      generateFeedback();
    }
  });