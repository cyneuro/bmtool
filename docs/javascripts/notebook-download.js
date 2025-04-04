document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a notebook page by looking for notebook-specific elements
  if (document.querySelector('.jp-Notebook') || 
      document.querySelector('.jp-RenderedHTMLCommon') || 
      document.querySelector('.jp-Cell')) {
    
    // Get the current URL path
    const currentPath = window.location.pathname;
    
    // URL structure is typically like:
    // /bmtool/site/examples/bmplot/bmplot/
    // We need to link to:
    // /bmtool/docs/examples/bmplot/bmplot.ipynb
    
    // Extract the relevant part of the path
    const pathSegments = currentPath.split('/').filter(segment => segment.length > 0);
    const notebookPath = pathSegments[pathSegments.length - 1] + '.ipynb';
    const categoryPath = pathSegments[pathSegments.length - 2];
    
    // Construct the full path to the original .ipynb file 
    // We need to use a relative path that goes from the site to the docs
    const downloadPath = `../../docs/examples/${categoryPath}/${notebookPath}`;
    
    // Create the download button
    const downloadButton = document.createElement('a');
    downloadButton.href = downloadPath;
    downloadButton.className = 'notebook-download-button';
    downloadButton.textContent = 'Download Notebook';
    downloadButton.download = notebookPath;
    downloadButton.title = 'Download ' + notebookPath;
    
    // Get the main content area and insert button
    const contentArea = document.querySelector('.md-content__inner');
    if (contentArea) {
      contentArea.appendChild(downloadButton);
    }
  }
}); 