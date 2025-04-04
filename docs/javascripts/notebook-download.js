document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a notebook page by looking for notebook-specific elements
  if (document.querySelector('.jp-Notebook') || 
      document.querySelector('.jp-RenderedHTMLCommon') || 
      document.querySelector('.jp-Cell')) {
    
    // Get the current URL path
    const currentPath = window.location.pathname;
    
    // Extract information from the path
    const pathSegments = currentPath.split('/').filter(segment => segment.length > 0);
    
    // From the logs, we can see the actual notebook paths are like:
    // /Users/gregglickert/Documents/GitHub/bmtool/docs/examples/notebooks/bmplot/bmplot.ipynb
    
    // Parse the path to get notebook info
    let notebookCategory = '';
    let notebookName = '';
    
    // Look for 'notebooks' in the path to find where the category starts
    if (pathSegments.includes('notebooks')) {
      const notebooksIndex = pathSegments.indexOf('notebooks');
      
      // From the logs we can see that notebooks are placed in directories like:
      // examples/notebooks/bmplot/bmplot/bmplot.ipynb
      // The structure is: category, then a folder with the same name, then the notebook
      
      // The category is the segment after 'notebooks'
      if (notebooksIndex + 1 < pathSegments.length) {
        notebookCategory = pathSegments[notebooksIndex + 1];
        
        // The notebook name is the same as the category in most cases
        notebookName = notebookCategory;
        
        // Special case for paths with deeper nesting like single_cell/Allen_tutorial
        if (notebooksIndex + 2 < pathSegments.length) {
          const subFolder = pathSegments[notebooksIndex + 2];
          
          // Check if we're in a deeper structure like single_cell/Allen_tutorial
          if (subFolder !== notebookCategory) {
            notebookCategory = `${notebookCategory}/${subFolder}`;
            notebookName = subFolder;
          }
          
          // Special case for the Allen tutorial which has another level
          if (notebooksIndex + 3 < pathSegments.length && 
              notebookCategory.includes('Allen_tutorial')) {
            notebookName = pathSegments[notebooksIndex + 3];
          }
        }
      }
    }
    
    if (notebookCategory && notebookName) {
      // Based on logs, the original source file is at:
      // docs/examples/notebooks/[category]/[name].ipynb
      const sourceUrl = `https://raw.githubusercontent.com/cyneuro/bmtool/master/docs/examples/notebooks/${notebookCategory}/${notebookName}.ipynb`;
      
      // Create download button
      const downloadButton = document.createElement('button');
      downloadButton.className = 'notebook-download-button';
      downloadButton.textContent = 'Download Notebook';
      downloadButton.title = `Download ${notebookName}.ipynb`;
      
      // Add click event to force download
      downloadButton.addEventListener('click', function() {
        // We need to fetch the content first
        fetch(sourceUrl)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to fetch notebook (${response.status} ${response.statusText})`);
            }
            return response.blob();
          })
          .then(blob => {
            // Create a blob URL and force download
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = `${notebookName}.ipynb`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            // Clean up the blob URL
            setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
          })
          .catch(error => {
            console.error('Error downloading notebook:', error);
            alert('Error downloading notebook. Please check console for details.');
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