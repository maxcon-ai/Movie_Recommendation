function toggleForm(selectedForm) {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => form.classList.remove('active'));
    document.getElementById(selectedForm).classList.add('active');
}
async function fetchSuggestions(query) {
    if (query.length < 2) return;
    const suggestionsList = document.getElementById('suggestions');
    suggestionsList.innerHTML = '';
    try {
        const response = await fetch(`/search?query=${query}`);
        if (!response.ok) throw new Error('Failed to fetch suggestions');
        const suggestions = await response.json();
        suggestions.forEach(movie => {
            const option = document.createElement('option');
            option.value = movie;
            suggestionsList.appendChild(option);
        });
    } catch (error) {
        console.error('Error fetching suggestions:', error);
    }
}