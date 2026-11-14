document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach((item) => setTimeout(() => item.remove(), 7000));
  const form = document.querySelector('#generator-form');
  if (form) form.addEventListener('submit', (event) => {
    const count = Number(form.elements.count.value);
    const customers = Number(form.elements.customers.value);
    const start = form.elements.start_date.value;
    const end = form.elements.end_date.value;
    const segments = [...form.querySelectorAll('input[name="segments"]:checked')];
    if (customers > count || !segments.length || start > end) {
      event.preventDefault();
      window.alert('Check customer count, select at least one segment, and ensure the start date is before the end date.');
    }
  });
  if (!window.Plotly) return;
  const layout = {paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#aeb8ce',family:'DM Sans'},margin:{l:40,r:15,t:10,b:35},xaxis:{gridcolor:'#26314a',zerolinecolor:'#26314a'},yaxis:{gridcolor:'#26314a',zerolinecolor:'#26314a'},showlegend:false};
  const read = (el, key) => JSON.parse(el.dataset[key] || '[]');
  const chart = (id, trace, extra={}) => {const el=document.getElementById(id);if(el) Plotly.newPlot(el,[trace],{...layout,...extra},{displayModeBar:false,responsive:true});};
  const category=document.getElementById('category-chart');
  if(category) chart('category-chart',{type:'bar',x:read(category,'labels'),y:read(category,'values'),marker:{color:'#8c6cff',line:{color:'#bb9fff',width:1}},hovertemplate:'%{x}<br>$%{y:,.2f}<extra></extra>'},{yaxis:{...layout.yaxis,tickprefix:'$'}});
  const fraud=document.getElementById('fraud-chart');
  if(fraud) chart('fraud-chart',{type:'pie',labels:['Normal','Suspicious'],values:[Number(fraud.dataset.normal),Number(fraud.dataset.fraud)],hole:.64,marker:{colors:['#45d6ca','#ff7197']},textinfo:'label+percent',textfont:{size:10}},{showlegend:true,legend:{orientation:'h',y:-.1}});
  const types=document.getElementById('fraud-types-chart');
  if(types) chart('fraud-types-chart',{type:'bar',x:read(types,'values'),y:read(types,'labels'),orientation:'h',marker:{color:'#ab79ff'}},{margin:{l:125,r:15,t:10,b:30},xaxis:{...layout.xaxis,dtick:1}});
  const spend=document.getElementById('spend-chart');
  if(spend) chart('spend-chart',{type:'bar',x:read(spend,'labels'),y:read(spend,'values'),marker:{color:'#45d6ca'}},{yaxis:{...layout.yaxis,tickprefix:'$'}});
  const red=document.getElementById('redemption-chart');
  if(red) chart('redemption-chart',{type:'scatter',mode:'lines+markers',x:read(red,'labels'),y:read(red,'values'),line:{color:'#9b7aff',width:3},marker:{color:'#45d6ca',size:7},fill:'tozeroy',fillcolor:'#906dff20'});
  const seg=document.getElementById('segment-chart');
  if(seg) chart('segment-chart',{type:'bar',x:read(seg,'labels'),y:read(seg,'values'),marker:{color:['#906dff','#45d6ca','#f2b862']}});
});
