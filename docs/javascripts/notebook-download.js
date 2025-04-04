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
    let nestedDir = '';
    
    // Check if we're in a notebook view
    if (pathSegments.includes('examples')) {
      const examplesIndex = pathSegments.indexOf('examples');
      if (examplesIndex < pathSegments.length - 1) {
        category = pathSegments[examplesIndex + 1];
        
        // Handle nested directories by joining all path segments after category
        const remainingSegments = pathSegments.slice(examplesIndex + 2);
        if (remainingSegments.length > 0) {
          notebookName = remainingSegments[remainingSegments.length - 1];
          
          // If there are segments between category and notebook, they form a nested directory
          if (remainingSegments.length > 1) {
            nestedDir = remainingSegments.slice(0, -1).join('/') + '/';
          }
        }
      }
    }
    
    // Use GitHub's raw content URL for direct file download
    const rawUrl = `https://raw.githubusercontent.com/cyneuro/bmtool/master/docs/examples/notebooks/${category}/${nestedDir}${notebookName}.ipynb`;
    
    // Create the download button only if we have a valid notebook name
    if (notebookName) {
      const downloadButton = document.createElement('button');
      downloadButton.className = 'notebook-download-button';
      downloadButton.textContent = 'Download Notebook';
      downloadButton.title = `Download ${notebookName}.ipynb`;
      
      // Add click event to handle the download
      downloadButton.addEventListener('click', function() {
        // Create a link to download the notebook
        const link = document.createElement('a');
        link.href = rawUrl;
        link.download = `${notebookName}.ipynb`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
      
      // Get the main content area and insert button
      const contentArea = document.querySelector('.md-content__inner');
      if (contentArea) {
        contentArea.appendChild(downloadButton);
      }
    }
  }
}); 