function showAnswer(question, answer) {
    document.getElementById('displayed-question').innerText = decodeURIComponent(question);
    document.getElementById('displayed-answer').innerText = decodeURIComponent(answer);
    document.getElementById('qa-block').style.display = 'block';

    // Show the rating section and set hidden inputs
    document.getElementById('rating-section').style.display = 'block';
    document.getElementById('hidden-question').value = question;
    document.getElementById('hidden-answer').value = answer;
}

// Add event listeners to all list items after the page has loaded
window.addEventListener('load', function() {
    // Get all list items
    var listItems = document.querySelectorAll('.sidebar ul li');

    // Attach the showAnswer function to each list item's click event
    listItems.forEach(function(item) {
        item.addEventListener('click', function() {
            var question = this.textContent;  // Get question directly from the list item
            var answer = this.getAttribute('data-answer');  // Get answer from a data attribute

            showAnswer(question, answer);
        });
    });
    // If there is a question and answer loaded from the server, show the rating section
    var hiddenQA = document.getElementById('hidden-qa');
    var question = hiddenQA.getAttribute('data-question');
    var answer = hiddenQA.getAttribute('data-answer');
    if (question && answer) {
        showAnswer(question, answer);
    }
});
