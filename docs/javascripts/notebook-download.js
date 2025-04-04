document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a notebook page by looking for notebook-specific elements
  if (document.querySelector('.jp-Notebook') || 
      document.querySelector('.jp-RenderedHTMLCommon') || 
      document.querySelector('.jp-Cell')) {
    
    // Get the current URL and replace .html with .ipynb to create download link
    const currentPath = window.location.pathname;
    const notebookPath = currentPath.replace(/\.html$/, '.ipynb');
    
    // Extract the notebook name from the path
    const notebookName = notebookPath.split('/').pop();
    
    // Create the download button
    const downloadButton = document.createElement('a');
    downloadButton.href = notebookPath;
    downloadButton.className = 'notebook-download-button';
    downloadButton.textContent = 'Download Notebook';
    downloadButton.download = notebookName;
    downloadButton.title = 'Download ' + notebookName;
    
    // Get the main content area and insert button at the top
    const contentArea = document.querySelector('.md-content__inner');
    if (contentArea) {
      contentArea.appendChild(downloadButton);
    }
  }
}); 