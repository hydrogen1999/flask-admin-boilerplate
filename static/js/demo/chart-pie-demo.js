// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

// Pie Chart Example
var ctx = document.getElementById("myPieChart");
var jsonData = $.ajax({
  url: 'percent.json',
  dataType: 'json',
}).done(function(results) {
  //get values that only needed
  // processedData = processData(results);
  // data = {
  //   labels: processedData.labels,
  //   datasets: [{
  //     label: "MSFT Stock - 2018",
  //     fillColor: "rgba(151,187,205,0.2)",
  //     strokeColor: "rgba(151,187,205,1)",
  //     pointColor: "rgba(151,187,205,1)",
  //     pointStrokeColor: "#fff",
  //     pointHighlightFill: "#fff",
  //     pointHighlightStroke: "rgba(151,187,205,1)",
  //     data: processedData.data
  //   }]
  // };

  var myPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: jsonData.index,
      datasets: [{
        data: jsonData.data,
        backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
        hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf'],
        hoverBorderColor: "rgba(234, 236, 244, 1)",
      }],
    },
    options: {
      maintainAspectRatio: false,
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#dddfeb',
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        caretPadding: 10,
      },
      legend: {
        display: false
      },
      cutoutPercentage: 80,
    },
  });
  
});
// var myPieChart = new Chart(ctx, {
//   type: 'doughnut',
//   data: {
//     labels: jsonData.index,
//     datasets: [{
//       data: jsonData.data,
//       backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
//       hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf'],
//       hoverBorderColor: "rgba(234, 236, 244, 1)",
//     }],
//   },
//   options: {
//     maintainAspectRatio: false,
//     tooltips: {
//       backgroundColor: "rgb(255,255,255)",
//       bodyFontColor: "#858796",
//       borderColor: '#dddfeb',
//       borderWidth: 1,
//       xPadding: 15,
//       yPadding: 15,
//       displayColors: false,
//       caretPadding: 10,
//     },
//     legend: {
//       display: false
//     },
//     cutoutPercentage: 80,
//   },
// });
