export type Detection = {label:string; confidence:number; x_min:number; y_min:number; x_max:number; y_max:number; latitude?:number; longitude?:number};
export type Frame = {filename:string; detections:Detection[]; image_width?:number; image_height?:number};
export function parseResults(raw:unknown):Frame[]{
 const list=Array.isArray(raw)?raw:(raw as {images?:unknown})?.images;
 if(!Array.isArray(list)||!list.length||list.length>10000) throw Error('Choose a predictions.json containing 1–10,000 images.');
 return list.map((f)=>{
  if(!f || typeof f.filename!=='string'||!f.filename||!Array.isArray(f.detections)) throw Error('Each image needs a filename and detections list.');
  if(f.detections.length>10000) throw Error('Too many detections in one image.');
  const detections=f.detections.map((d:Detection)=>{
   if(!d||typeof d.label!=='string'||![d.confidence,d.x_min,d.y_min,d.x_max,d.y_max].every(Number.isFinite)||d.confidence<0||d.confidence>1||d.x_min<0||d.y_min<0||d.x_max<=d.x_min||d.y_max<=d.y_min) throw Error('Invalid detection coordinates or confidence.');
   const geo=d.latitude!==undefined||d.longitude!==undefined;
   if(geo&&(!Number.isFinite(d.latitude)||!Number.isFinite(d.longitude)||Math.abs(d.latitude!)>90||Math.abs(d.longitude!)>180)) throw Error('Invalid latitude / longitude.');
   return {...d};
  });
  return {filename:f.filename,detections,image_width:f.image_width,image_height:f.image_height};
 });
}
export function csvReport(frames:Frame[],threshold:number){
 const quote=(v:unknown)=>{let s=String(v);if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'};
 return [['filename','label','confidence','x_min','y_min','x_max','y_max','latitude','longitude'].join(','),...frames.flatMap(f=>f.detections.filter(d=>d.confidence>=threshold).map(d=>[f.filename,d.label,d.confidence,d.x_min,d.y_min,d.x_max,d.y_max,d.latitude??'',d.longitude??''].map(quote).join(',')))].join('\r\n');
}
