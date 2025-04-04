document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a notebook page by looking for notebook-specific elements
  if (document.querySelector('.jp-Notebook') || 
      document.querySelector('.jp-RenderedHTMLCommon') || 
      document.querySelector('.jp-Cell')) {
    
    // Get the current URL path
    const currentPath = window.location.pathname;
    
    // Extract information from the path
    const pathSegments = currentPath.split('/').filter(segment => segment.length > 0);
    
    // Find the notebook name and category from the path
    let notebookName = '';
    let category = '';
    
    // Check if we're in a notebook view
    if (pathSegments.includes('examples')) {
      const examplesIndex = pathSegments.indexOf('examples');
      if (examplesIndex < pathSegments.length - 1) {
        category = pathSegments[examplesIndex + 1];
        if (examplesIndex + 2 < pathSegments.length) {
          notebookName = pathSegments[examplesIndex + 2];
        }
      }
    }
    
    // Create the GitHub raw content URL with the correct path format
    const rawUrl = `https://raw.githubusercontent.com/cyneuro/bmtool/refs/heads/master/docs/examples/${category}/${notebookName}.ipynb`;
    
    // Create the download button only if we have a valid notebook name
    if (notebookName) {
      const downloadButton = document.createElement('button');
      downloadButton.className = 'notebook-download-button';
      downloadButton.textContent = 'Download Notebook';
      downloadButton.title = `Download ${notebookName}.ipynb`;
      
      // Add click event to handle the download
      downloadButton.addEventListener('click', function() {
        // Use fetch to get the raw file content
        fetch(rawUrl)
          .then(response => response.blob())
          .then(blob => {
            // Create a temporary anchor element
            const link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = `${notebookName}.ipynb`;
            link.style.display = 'none';
            
            // Append to the document, click and remove
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          })
          .catch(error => {
            console.error('Error downloading notebook:', error);
            alert('Error downloading notebook. Please try again later or contact maintainers.');
          });
      });
      
      // Get the main content area and insert button
      const contentArea = document.querySelector('.md-content__inner');
      if (contentArea) {
        contentArea.appendChild(downloadButton);
      }
    }
  }
}); 